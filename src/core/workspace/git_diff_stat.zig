const std = @import("std");
const io_mod = @import("../shared/io.zig");

const Allocator = std.mem.Allocator;

// ponytail: synchronous git subprocess, gated by this interval so a render
// triggered by every keystroke doesn't spawn a process every frame. Move to
// a background thread (like model_cache_runtime's refresh pattern) if this
// throttle is ever observed to stall the UI on very large repositories.
const refresh_interval_ms: i64 = 2000;
const timeout_ms: i64 = 1500;
const output_limit_bytes: usize = 4096;

/// Total insertions+deletions in the working tree vs HEAD, refreshed at most
/// once per `refresh_interval_ms`. Caches across workspace-root changes.
pub const Runtime = struct {
    enabled: bool = false,
    workspace_root: []u8 = &.{},
    lines_changed: u32 = 0,
    last_refresh_ms: i64 = std.math.minInt(i64),

    pub fn deinit(self: *Runtime, alloc: Allocator) void {
        if (self.workspace_root.len > 0) alloc.free(self.workspace_root);
        self.* = .{};
    }

    /// Returns the cached line count, refreshing it first when due. Never
    /// errors: any failure (no git, not a repo, timeout) just clears the count.
    pub fn refresh(self: *Runtime, alloc: Allocator, workspace_root: []const u8) u32 {
        if (!self.enabled or workspace_root.len == 0) return 0;
        if (!std.mem.eql(u8, self.workspace_root, workspace_root)) {
            const next_root = alloc.dupe(u8, workspace_root) catch return self.lines_changed;
            if (self.workspace_root.len > 0) alloc.free(self.workspace_root);
            self.workspace_root = next_root;
            self.lines_changed = 0;
            self.last_refresh_ms = std.math.minInt(i64);
        }

        const now = io_mod.milliTimestamp();
        if (now - self.last_refresh_ms < refresh_interval_ms) return self.lines_changed;
        self.last_refresh_ms = now;
        self.lines_changed = fetchLinesChanged(alloc, self.workspace_root) orelse 0;
        return self.lines_changed;
    }
};

fn fetchLinesChanged(alloc: Allocator, workspace_root: []const u8) ?u32 {
    const result = std.process.run(alloc, io_mod.getIo(), .{
        .argv = &.{ "git", "diff", "HEAD", "--shortstat" },
        .cwd = .{ .path = workspace_root },
        .timeout = .{ .duration = .{ .clock = .awake, .raw = .fromMilliseconds(timeout_ms) } },
        .stdout_limit = .limited(output_limit_bytes),
        .stderr_limit = .limited(output_limit_bytes),
    }) catch return null;
    defer alloc.free(result.stdout);
    defer alloc.free(result.stderr);
    if (result.term != .exited or result.term.exited != 0) return null;
    return parseShortstat(result.stdout);
}

/// Parses git's `--shortstat` line, e.g.
/// " 3 files changed, 42 insertions(+), 7 deletions(-)".
fn parseShortstat(text: []const u8) u32 {
    var total: u32 = 0;
    var pending: ?u32 = null;
    var tokens = std.mem.tokenizeAny(u8, text, " \t\n\r,");
    while (tokens.next()) |token| {
        if (std.mem.startsWith(u8, token, "insertion") or std.mem.startsWith(u8, token, "deletion")) {
            if (pending) |value| total += value;
            pending = null;
        } else {
            pending = std.fmt.parseUnsigned(u32, token, 10) catch null;
        }
    }
    return total;
}

test "shortstat parsing sums insertions and deletions" {
    try std.testing.expectEqual(@as(u32, 49), parseShortstat(" 3 files changed, 42 insertions(+), 7 deletions(-)\n"));
    try std.testing.expectEqual(@as(u32, 5), parseShortstat(" 1 file changed, 5 insertions(+)\n"));
    try std.testing.expectEqual(@as(u32, 1), parseShortstat(" 1 file changed, 1 deletion(-)\n"));
    try std.testing.expectEqual(@as(u32, 0), parseShortstat(""));
    try std.testing.expectEqual(@as(u32, 0), parseShortstat("garbage text with no numbers"));
}

test "git diff runtime is disabled by default and reports zero with no workspace" {
    var runtime: Runtime = .{};
    defer runtime.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 0), runtime.refresh(std.testing.allocator, ""));

    runtime.enabled = true;
    try std.testing.expectEqual(@as(u32, 0), runtime.refresh(std.testing.allocator, ""));
}
