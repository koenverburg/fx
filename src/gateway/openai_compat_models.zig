const std = @import("std");
const credentials = @import("../core/auth/credentials.zig");
const model_catalog = @import("../core/gateway/model_catalog.zig");
const gateway_provider = @import("../core/gateway/gateway_provider.zig");
const io_mod = @import("../core/shared/io.zig");
const secret = @import("../core/auth/secret.zig");
const gateway_client = @import("client.zig");

const base_url_env = "FX_OPENAI_COMPAT_BASE_URL";
const api_key_env = "FX_OPENAI_COMPAT_API_KEY";
const models_path = "/v1/models";
const max_catalog_bytes: usize = 1024 * 1024;
const max_catalog_models: usize = 512;
const fetch_timeout_ms: i64 = 30_000;

pub const model_catalog_provider = model_catalog.Provider{
    .fetch_fn = fetchCatalogForProvider,
    .provider_id = .openai_compat,
};

pub const cli_model_catalog_provider = gateway_provider.CliModelCatalogProvider{
    .fetch_fn = fetchCliModelCatalog,
};

/// A local server has no login: every access level (host-managed or public)
/// can list its models.
fn fetchCliModelCatalog(
    _: ?*anyopaque,
    alloc: std.mem.Allocator,
    input: gateway_provider.CliModelCatalogInput,
) gateway_provider.CliModelCatalogResult {
    return switch (model_catalog.fetchWithPublicFallback(model_catalog_provider, alloc, .{
        .access = input.access,
        .endpoint = input.endpoint,
        .cancel_flag = input.cancel_flag,
        .view = .full,
    })) {
        .loaded => |loaded| blk: {
            var catalog = loaded.catalog;
            defer model_catalog.freeModelCatalog(alloc, &catalog);
            const ids = model_catalog.projectModelIds(alloc, catalog.items) catch return .{ .failure = .{
                .access = loaded.provenance.access,
                .anonymous_fallback_used = false,
                .failure = .{ .category = .resource_exhausted },
            } };
            break :blk .{ .loaded = .{ .ids = ids, .provenance = loaded.provenance } };
        },
        .failed => |failure| .{ .failure = failure },
    };
}

fn fetchCatalogForProvider(
    _: ?*anyopaque,
    alloc: std.mem.Allocator,
    _: model_catalog.FetchInput,
) std.mem.Allocator.Error!model_catalog.ProviderResult {
    const request_url = modelsUrl(alloc) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return .{ .failure = .{ .category = .runtime } };
    };
    defer alloc.free(request_url);

    var fallback_cancel = std.atomic.Value(bool).init(false);
    const deadline = std.Io.Clock.Timestamp.fromNow(io_mod.getIo(), .{
        .clock = .awake,
        .raw = .fromMilliseconds(fetch_timeout_ms),
    });
    var response = fetchCatalogResponse(alloc, request_url, &fallback_cancel, deadline) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return .{ .failure = catalogFetchFailure(err) };
    };
    defer response.deinit(alloc);
    if (response.status != .ok) {
        return .{ .failure = model_catalog.failureForHttpStatus(response.status) };
    }
    const catalog = parseCatalog(alloc, response.body) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return .{ .failure = .{ .category = .malformed_response, .http_status = .ok } };
    };
    return .{ .catalog = catalog };
}

fn catalogFetchFailure(err: anyerror) model_catalog.Failure {
    if (err == error.Cancelled) return .{ .category = .cancellation };
    if (err == error.OpenAiCompatBaseUrlMissing) return .{ .category = .runtime };
    if (err == error.OpenAiCompatModelCatalogTooLarge) return .{ .category = .malformed_response };
    return .{ .category = .transport, .retryable = true };
}

const FetchResponse = struct {
    status: std.http.Status,
    body: []u8,

    pub fn deinit(self: *FetchResponse, alloc: std.mem.Allocator) void {
        secret.zeroAndFree(alloc, self.body);
        self.* = undefined;
    }
};

const FetchOperation = struct {
    alloc: std.mem.Allocator,
    url: []const u8,

    pub fn run(self: *@This()) !FetchResponse {
        var client: std.http.Client = .{ .allocator = self.alloc, .io = io_mod.getIo() };
        defer client.deinit();
        const configured_key = io_mod.getenv(api_key_env);
        const auth_header: ?[]u8 = if (configured_key) |key| (if (key.len > 0)
            try std.fmt.allocPrint(self.alloc, "Bearer {s}", .{key})
        else
            null) else null;
        defer if (auth_header) |value| secret.zeroAndFree(self.alloc, value);
        var headers: std.http.Client.Request.Headers = .{
            .user_agent = .{ .override = gateway_client.user_agent },
            .accept_encoding = .omit,
        };
        if (auth_header) |value| headers.authorization = .{ .override = value };
        const body_buffer = try self.alloc.alloc(u8, max_catalog_bytes + 1);
        defer secret.zeroAndFree(self.alloc, body_buffer);
        var response_writer = std.Io.Writer.fixed(body_buffer);
        const extra_headers = [_]std.http.Header{.{ .name = "accept", .value = "application/json" }};
        const result = client.fetch(.{
            .location = .{ .url = self.url },
            .method = .GET,
            .headers = headers,
            .extra_headers = &extra_headers,
            .response_writer = &response_writer,
            .redirect_behavior = .unhandled,
        }) catch |err| switch (err) {
            error.WriteFailed => return error.OpenAiCompatModelCatalogTooLarge,
            else => return err,
        };
        const body = response_writer.buffered();
        return .{ .status = result.status, .body = try self.alloc.dupe(u8, body) };
    }
};

fn fetchCatalogResponse(
    alloc: std.mem.Allocator,
    url: []const u8,
    cancel_flag: *std.atomic.Value(bool),
    deadline: std.Io.Clock.Timestamp,
) !FetchResponse {
    var operation = FetchOperation{ .alloc = alloc, .url = url };
    return gateway_client.runBoundedHttpOperation(FetchResponse, alloc, cancel_flag, deadline, &operation);
}

fn modelsUrl(alloc: std.mem.Allocator) ![]u8 {
    const configured = io_mod.getenv(base_url_env) orelse return error.OpenAiCompatBaseUrlMissing;
    const trimmed = std.mem.trimEnd(u8, configured, "/");
    return std.fmt.allocPrint(alloc, "{s}{s}", .{ trimmed, models_path });
}

fn parseCatalog(alloc: std.mem.Allocator, body: []const u8) !std.ArrayList(model_catalog.ModelCatalogEntry) {
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    if (parsed.value != .object) return error.OpenAiCompatMalformedCatalog;
    const data = parsed.value.object.get("data") orelse return error.OpenAiCompatMalformedCatalog;
    if (data != .array) return error.OpenAiCompatMalformedCatalog;
    if (data.array.items.len > max_catalog_models) return error.OpenAiCompatModelCatalogTooLarge;

    var entries: std.ArrayList(model_catalog.ModelCatalogEntry) = .empty;
    errdefer model_catalog.freeModelCatalog(alloc, &entries);
    for (data.array.items) |item| {
        if (item != .object) continue;
        const id_value = item.object.get("id") orelse continue;
        if (id_value != .string or id_value.string.len == 0) continue;
        const id = try alloc.dupe(u8, id_value.string);
        errdefer alloc.free(id);
        const model_type = try alloc.dupe(u8, "chat");
        errdefer alloc.free(model_type);
        try entries.append(alloc, .{ .id = id, .model_type = model_type, .has_tool_use = true });
    }
    return entries;
}

test "openai-compatible catalog parses model ids from a v1/models response" {
    const body = "{\"data\":[{\"id\":\"mellum2\",\"object\":\"model\"},{\"id\":\"\"},{\"other\":1}]}";
    var entries = try parseCatalog(std.testing.allocator, body);
    defer model_catalog.freeModelCatalog(std.testing.allocator, &entries);
    try std.testing.expectEqual(@as(usize, 1), entries.items.len);
    try std.testing.expectEqualStrings("mellum2", entries.items[0].id);
}
