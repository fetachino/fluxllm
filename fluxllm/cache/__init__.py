"""Cache package: block manager and prefix cache.

The block manager implements *logical* paged KV-cache management: fixed-size
blocks, refcounted ownership, free tracking, and capacity enforcement. It
does not implement attention-kernel-level paged memory (see docs/kv-cache.md
and limitations in the README).
"""
