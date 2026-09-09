const std = @import("std");
const debug_trace = @import("../core/shared/debug_trace.zig");
const stream_provider = @import("../core/agent/stream_provider.zig");
const io_mod = @import("../core/shared/io.zig");
const types = @import("../core/shared/types.zig");
const gateway_client = @import("client.zig");
const sse_stream = @import("sse.zig");

const Allocator = std.mem.Allocator;

/// Base URL of the OpenAI-compatible server (llama.cpp, mlx_lm.server, ...).
/// No default: a local inference server has no canonical address.
const base_url_env = "FX_OPENAI_COMPAT_BASE_URL";
/// Optional: most local servers need no key. Empty or unset sends no Authorization header.
const api_key_env = "FX_OPENAI_COMPAT_API_KEY";
const chat_completions_path = "/v1/chat/completions";
const max_error_body_bytes: usize = 256 * 1024;
const max_sse_line_bytes: usize = 1024 * 1024;
const max_sse_aggregate_bytes: usize = 64 * 1024 * 1024;
const max_sse_events: usize = 200_000;
const max_content_bytes: usize = 16 * 1024 * 1024;
const transfer_buffer_bytes: usize = 256 * 1024;
const connect_timeout_ms: i64 = 30_000;

pub const agent_stream_provider = stream_provider.Provider{
    .stream_fn = streamCompletion,
    .build_request_fn = buildRequestForProvider,
};

fn validateModel(model: []const u8) !void {
    if (model.len == 0 or model.len > 256) return error.InvalidOpenAiCompatModel;
    for (model) |byte| {
        if (byte <= 0x20 or byte == 0x7f) return error.InvalidOpenAiCompatModel;
    }
}

/// Chat Completions has no distinct instructions/messages lanes: instructions
/// become leading system messages, ahead of the conversation.
pub fn buildRequest(
    alloc: Allocator,
    request: stream_provider.RequestData,
) ![]u8 {
    try request.validatePrompt();
    try validateModel(request.model);

    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    const writer = &out.writer;
    try writer.writeAll("{\"model\":");
    try std.json.Stringify.value(request.model, .{}, writer);
    try writer.writeAll(",\"stream\":true,\"stream_options\":{\"include_usage\":true},\"messages\":[");
    var wrote_message = false;
    for (request.instructions) |instruction| {
        const text = instruction.content orelse continue;
        if (text.len == 0) continue;
        if (wrote_message) try writer.writeByte(',');
        try writeMessage(writer, "system", text, null);
        wrote_message = true;
    }
    for (request.messages) |message| {
        if (wrote_message) try writer.writeByte(',');
        try writeMessage(writer, @tagName(message.role), message.content orelse "", message.tool_call_id);
        wrote_message = true;
    }
    try writer.writeByte(']');
    if (request.max_output_tokens) |limit| try writer.print(",\"max_tokens\":{d}", .{limit});
    try writer.writeByte('}');
    return out.toOwnedSlice();
}

fn writeMessage(writer: *std.Io.Writer, role: []const u8, content: []const u8, tool_call_id: ?[]const u8) !void {
    try writer.writeAll("{\"role\":");
    try std.json.Stringify.value(role, .{}, writer);
    try writer.writeAll(",\"content\":");
    try std.json.Stringify.value(content, .{}, writer);
    if (tool_call_id) |id| {
        try writer.writeAll(",\"tool_call_id\":");
        try std.json.Stringify.value(id, .{}, writer);
    }
    try writer.writeByte('}');
}

fn buildRequestForProvider(
    _: ?*anyopaque,
    alloc: Allocator,
    request: stream_provider.RequestData,
) anyerror![]u8 {
    return buildRequest(alloc, request);
}

fn resolvedEndpoint(alloc: Allocator) ![]u8 {
    const configured = io_mod.getenv(base_url_env) orelse return error.OpenAiCompatBaseUrlMissing;
    const trimmed = std.mem.trimEnd(u8, configured, "/");
    return std.fmt.allocPrint(alloc, "{s}{s}", .{ trimmed, chat_completions_path });
}

fn streamCompletion(
    _: ?*anyopaque,
    alloc: Allocator,
    request: stream_provider.ModelRequest,
) !stream_provider.Result {
    if (request.cancel_flag.load(.seq_cst)) return stream_provider.failResult(error.Cancelled);
    try validateModel(request.model);
    const payload = request.prepared_request_body orelse
        try buildRequest(alloc, request.data());
    defer if (request.prepared_request_body == null) alloc.free(payload);
    var result = streamPrepared(alloc, request, payload) catch |err| {
        if (request.cancel_flag.load(.seq_cst)) return stream_provider.failResult(error.Cancelled);
        if (requestDeadlineExpired(request)) return stream_provider.failResult(error.Timeout);
        request.attempt_evidence.network_failure = gateway_client.networkFailureEvidence(err, request.delivery.load());
        return err;
    };
    if (requestDeadlineExpired(request)) {
        result.deinit(alloc);
        return stream_provider.failResult(error.Timeout);
    }
    return result;
}

fn requestDeadlineExpired(request: stream_provider.ModelRequest) bool {
    const deadline = request.deadline orelse return false;
    const now = std.Io.Clock.Timestamp.now(io_mod.getIo(), .awake);
    return !std.Io.Clock.Timestamp.compare(now, .lt, deadline);
}

const OpenedRequest = struct {
    request: ?std.http.Client.Request,

    pub fn deinit(self: *OpenedRequest, _: Allocator) void {
        if (self.request) |*request| request.deinit();
        self.request = null;
    }

    pub fn take(self: *OpenedRequest) std.http.Client.Request {
        const request = self.request.?;
        self.request = null;
        return request;
    }
};

const OpenRequestOperation = struct {
    client: *std.http.Client,
    uri: std.Uri,
    auth_header: ?[]const u8,
    extra_headers: []const std.http.Header,

    pub fn run(self: *@This()) !OpenedRequest {
        var headers: std.http.Client.Request.Headers = .{
            .content_type = .{ .override = "application/json" },
            .accept_encoding = .omit,
            .user_agent = .{ .override = gateway_client.user_agent },
        };
        if (self.auth_header) |authorization| {
            headers.authorization = .{ .override = authorization };
        }
        return .{ .request = try self.client.request(.POST, self.uri, .{
            .headers = headers,
            .extra_headers = self.extra_headers,
            .keep_alive = false,
            .redirect_behavior = .unhandled,
        }) };
    }
};

pub fn streamPrepared(
    alloc: Allocator,
    request: stream_provider.ModelRequest,
    payload: []const u8,
) !stream_provider.Result {
    if (request.cancel_flag.load(.seq_cst)) return stream_provider.failResult(error.Cancelled);
    const request_endpoint = resolvedEndpoint(alloc) catch |err| switch (err) {
        error.OpenAiCompatBaseUrlMissing => return stream_provider.failResult(error.OpenAiCompatBaseUrlMissing),
        else => return err,
    };
    defer alloc.free(request_endpoint);
    const uri = try std.Uri.parse(request_endpoint);

    const configured_key = io_mod.getenv(api_key_env);
    const auth_header: ?[]u8 = if (configured_key) |key| (if (key.len > 0)
        try std.fmt.allocPrint(alloc, "Bearer {s}", .{key})
    else
        null) else null;
    defer if (auth_header) |value| alloc.free(value);

    var extra_headers_buf: [1]std.http.Header = .{
        .{ .name = "accept", .value = "text/event-stream" },
    };

    var client: std.http.Client = .{ .allocator = alloc, .io = io_mod.getIo() };
    defer client.deinit();
    var open_operation = OpenRequestOperation{
        .client = &client,
        .uri = uri,
        .auth_header = auth_header,
        .extra_headers = &extra_headers_buf,
    };
    var connect_deadline = std.Io.Clock.Timestamp.fromNow(io_mod.getIo(), .{
        .clock = .awake,
        .raw = .fromMilliseconds(connect_timeout_ms),
    });
    if (request.deadline) |deadline| {
        if (std.Io.Clock.Timestamp.compare(deadline, .lt, connect_deadline)) {
            connect_deadline = deadline;
        }
    }
    try request.admission.admit();
    var opened = try gateway_client.runBoundedHttpOperation(
        OpenedRequest,
        alloc,
        request.cancel_flag,
        connect_deadline,
        &open_operation,
    );
    var http_request = opened.take();
    defer http_request.deinit();
    var cancel_watch_done = std.atomic.Value(bool).init(false);
    const cancel_watcher = if (http_request.connection) |connection|
        if (request.deadline) |deadline|
            try gateway_client.spawnHttpCancelWatcherBounded(
                &cancel_watch_done,
                request.cancel_flag,
                deadline,
                connection.stream_writer.stream,
            )
        else
            try gateway_client.spawnHttpCancelWatcher(
                &cancel_watch_done,
                request.cancel_flag,
                connection.stream_writer.stream,
            )
    else
        null;
    defer {
        cancel_watch_done.store(true, .seq_cst);
        if (cancel_watcher) |thread| thread.join();
    }
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;

    http_request.transfer_encoding = .{ .content_length = payload.len };
    var send_buffer: [8192]u8 = undefined;
    request.delivery.markPossiblySent();
    var body_writer = try http_request.sendBodyUnflushed(&send_buffer);
    try body_writer.writer.writeAll(payload);
    try body_writer.end();
    if (http_request.connection) |connection| try connection.flush();
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;

    var response = try http_request.receiveHead(&.{});
    if (response.head.status != .ok) {
        var transfer: [16 * 1024]u8 = undefined;
        const reader = response.reader(&transfer);
        const bounded_body = reader.allocRemaining(alloc, .limited(max_error_body_bytes + 1)) catch |err| switch (err) {
            error.StreamTooLong => try alloc.dupe(u8, "OpenAI-compatible server error response exceeded the local limit"),
            else => return err,
        };
        const body = if (bounded_body.len > max_error_body_bytes) body: {
            alloc.free(bounded_body);
            break :body try alloc.dupe(u8, "OpenAI-compatible server error response exceeded the local limit");
        } else bounded_body;
        return .{ .failed = .{
            .kind = failureKind(response.head.status),
            .detail = body,
            .ownership = .owned,
        } };
    }

    var transfer_buffer: [transfer_buffer_bytes]u8 = undefined;
    const reader = response.reader(&transfer_buffer);
    const completion = try consumeSse(alloc, reader, request.events, request.cancel_flag, request.content_capture_limit);
    errdefer {
        var owned = stream_provider.Result{ .completed = .{
            .completion = completion,
            .ownership = .owned,
        } };
        owned.deinit(alloc);
    }
    return .{ .completed = .{
        .completion = completion,
        .usage = .{ .unavailable = .unbilled },
        .ownership = .owned,
    } };
}

fn failureKind(status: std.http.Status) stream_provider.FailureKind {
    return switch (status) {
        .bad_request => .invalid_request,
        .unauthorized => .unauthorized,
        .forbidden => .forbidden,
        .payload_too_large => .request_too_large,
        .too_many_requests => .rate_limited,
        .internal_server_error => .server_error,
        .bad_gateway => .bad_gateway,
        .service_unavailable => .unavailable,
        .gateway_timeout => .gateway_timeout,
        else => .provider_error,
    };
}

/// One `choices[0].delta` payload from a Chat Completions stream chunk.
/// Every field is an owned copy: the source `std.json.Value` strings live in
/// `parsed`'s arena (or borrow the SSE reader's reused line buffer), both of
/// which are gone by the time the caller uses this value.
const Delta = struct {
    content: ?[]u8 = null,
    reasoning_content: ?[]u8 = null,
    finish_reason: ?[]u8 = null,

    fn deinit(self: *Delta, alloc: Allocator) void {
        if (self.content) |value| alloc.free(value);
        if (self.reasoning_content) |value| alloc.free(value);
        if (self.finish_reason) |value| alloc.free(value);
        self.* = .{};
    }
};

fn parseChunk(alloc: Allocator, json_text: []const u8) !?Delta {
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, json_text, .{ .ignore_unknown_fields = true }) catch
        return error.InvalidOpenAiCompatSseEvent;
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return error.InvalidOpenAiCompatSseEvent;
    const choices = root.object.get("choices") orelse return null;
    if (choices != .array or choices.array.items.len == 0) return null;
    const choice = choices.array.items[0];
    if (choice != .object) return null;
    var delta: Delta = .{};
    errdefer delta.deinit(alloc);
    if (choice.object.get("finish_reason")) |value| {
        if (value == .string) delta.finish_reason = try alloc.dupe(u8, value.string);
    }
    const delta_value = choice.object.get("delta") orelse return delta;
    if (delta_value != .object) return delta;
    if (delta_value.object.get("content")) |value| {
        if (value == .string) delta.content = try alloc.dupe(u8, value.string);
    }
    if (delta_value.object.get("reasoning_content")) |value| {
        if (value == .string) delta.reasoning_content = try alloc.dupe(u8, value.string);
    }
    return delta;
}

fn consumeSse(
    alloc: Allocator,
    stream_reader: *std.Io.Reader,
    events: stream_provider.EventSink,
    cancel_flag: *std.atomic.Value(bool),
    content_capture_limit: ?usize,
) !types.ModelCompletion {
    var sse: sse_stream.Reader = .{ .max_event_bytes = max_sse_line_bytes, .max_total_bytes = max_sse_aggregate_bytes };
    defer sse.deinit(alloc);
    var content: std.ArrayList(u8) = .empty;
    errdefer content.deinit(alloc);
    var finish_reason: ?types.ProviderFinishReason = null;
    var event_count: usize = 0;
    const capture_limit = content_capture_limit orelse max_content_bytes;

    while (sse.next(alloc, stream_reader, cancel_flag) catch |err| return mapSseError(err)) |json_text| {
        if (std.mem.eql(u8, json_text, "[DONE]")) break;
        event_count += 1;
        if (event_count > max_sse_events) return error.OpenAiCompatResourceLimitExceeded;
        var delta = (try parseChunk(alloc, json_text)) orelse continue;
        defer delta.deinit(alloc);
        if (delta.content) |chunk| {
            if (chunk.len > 0) {
                events.emit(.{ .content_delta = chunk });
                if (content.items.len + chunk.len <= capture_limit) {
                    try content.appendSlice(alloc, chunk);
                }
            }
        }
        if (delta.reasoning_content) |chunk| {
            if (chunk.len > 0) events.emit(.{ .reasoning_delta = chunk });
        }
        if (delta.finish_reason) |raw| {
            finish_reason = types.ProviderFinishReason.parse_legacy(raw);
        }
    }

    return .{
        .content = if (content.items.len > 0) try content.toOwnedSlice(alloc) else null,
        .finish_reason = finish_reason,
    };
}

fn mapSseError(err: anyerror) anyerror {
    return switch (err) {
        error.EventTooLarge => error.OpenAiCompatSseEventTooLarge,
        error.StreamTooLarge => error.OpenAiCompatResourceLimitExceeded,
        else => err,
    };
}

test "openai-compatible request flattens instructions and messages into chat completions shape" {
    const instructions = [_]types.ChatMessage{.{ .role = .system, .content = "Be concise." }};
    const messages = [_]types.ChatMessage{
        .{ .role = .user, .content = "hello" },
    };
    const body = try buildRequest(std.testing.allocator, .{
        .model = "mellum2",
        .instructions = &instructions,
        .messages = &messages,
        .tool_choice = .none,
        .provider_options = .{},
    });
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.indexOf(u8, body, "\"model\":\"mellum2\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"stream\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "{\"role\":\"system\",\"content\":\"Be concise.\"}") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "{\"role\":\"user\",\"content\":\"hello\"}") != null);
}

test "openai-compatible sse reducer accumulates content and reasoning separately" {
    const sse_text =
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" ++
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n\n" ++
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n" ++
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n" ++
        "data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}]}\n\n" ++
        "data: [DONE]\n\n";
    var reader: std.Io.Reader = .fixed(sse_text);
    var cancelled = std.atomic.Value(bool).init(false);
    const Capture = struct {
        content: std.ArrayList(u8) = .empty,
        reasoning: std.ArrayList(u8) = .empty,

        fn emit(raw: *anyopaque, event: stream_provider.Event) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            switch (event) {
                .content_delta => |chunk| self.content.appendSlice(std.testing.allocator, chunk) catch unreachable,
                .reasoning_delta => |chunk| self.reasoning.appendSlice(std.testing.allocator, chunk) catch unreachable,
                else => {},
            }
        }
    };
    var capture: Capture = .{};
    defer capture.content.deinit(std.testing.allocator);
    defer capture.reasoning.deinit(std.testing.allocator);
    const completion = try consumeSse(
        std.testing.allocator,
        &reader,
        .{ .context = &capture, .emit_fn = Capture.emit },
        &cancelled,
        null,
    );
    defer if (completion.content) |value| std.testing.allocator.free(@constCast(value));

    try std.testing.expectEqualStrings("hello", capture.content.items);
    try std.testing.expectEqualStrings("thinking", capture.reasoning.items);
    try std.testing.expectEqualStrings("hello", completion.content.?);
    try std.testing.expectEqual(types.ProviderFinishReason.stop, completion.finish_reason.?);
}
