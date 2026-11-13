"""Short ragged GDN scan: read immutable source, emit every position's state."""
import torch
import triton
import triton.language as tl


@triton.jit
def _verify_kernel(X, G, BETA, WEIGHT, CONV, SSM, BOUNDS, SOURCES, DESTS, OUT,
                   CHANNELS: tl.constexpr, KH: tl.constexpr, VH: tl.constexpr,
                   DK: tl.constexpr, DV: tl.constexpr, KERNEL: tl.constexpr,
                   MAX_T: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    request, head = tl.program_id(0), tl.program_id(1)
    key_head = head // (VH // KH)
    source = tl.load(SOURCES + request)
    begin = tl.load(BOUNDS + request)
    end = tl.load(BOUNDS + request + 1)
    kk, vv = tl.arange(0, BK), tl.arange(0, BV)
    qch = key_head * DK + kk
    kch = KH * DK + key_head * DK + kk
    vch = 2 * KH * DK + head * DV + vv
    state_offsets = head * DK * DV + kk[:, None] * DV + vv[None, :]
    state = tl.load(SSM + source * VH * DK * DV + state_offsets,
                    (kk[:, None] < DK) & (vv[None, :] < DV), 0).to(tl.float32)
    for t in range(end - begin):
        cq = tl.full((BK,), 0, tl.float32)
        ck = tl.full((BK,), 0, tl.float32)
        cv = tl.full((BV,), 0, tl.float32)
        for j in tl.static_range(KERNEL):
            relative = t + j - (KERNEL - 1)
            old = t + j
            xq = tl.load(X + (begin + relative) * CHANNELS + qch, (relative >= 0) & (kk < DK), 0)
            xk = tl.load(X + (begin + relative) * CHANNELS + kch, (relative >= 0) & (kk < DK), 0)
            xv = tl.load(X + (begin + relative) * CHANNELS + vch, (relative >= 0) & (vv < DV), 0)
            if KERNEL > 1:
                hq = tl.load(CONV + (source * CHANNELS + qch) * (KERNEL-1) + old,
                             (relative < 0) & (kk < DK), 0)
                hk = tl.load(CONV + (source * CHANNELS + kch) * (KERNEL-1) + old,
                             (relative < 0) & (kk < DK), 0)
                hv = tl.load(CONV + (source * CHANNELS + vch) * (KERNEL-1) + old,
                             (relative < 0) & (vv < DV), 0)
                xq, xk, xv = tl.where(relative < 0, hq, xq), tl.where(relative < 0, hk, xk), tl.where(relative < 0, hv, xv)
            cq += xq.to(tl.float32) * tl.load(WEIGHT + qch*KERNEL+j, kk < DK, 0).to(tl.float32)
            ck += xk.to(tl.float32) * tl.load(WEIGHT + kch*KERNEL+j, kk < DK, 0).to(tl.float32)
            cv += xv.to(tl.float32) * tl.load(WEIGHT + vch*KERNEL+j, vv < DV, 0).to(tl.float32)
        # Match the convolution-output and SiLU BF16 roundings of the reference.
        cq, ck, cv = cq.to(X.dtype.element_ty).to(tl.float32), ck.to(X.dtype.element_ty).to(tl.float32), cv.to(X.dtype.element_ty).to(tl.float32)
        q = (cq * tl.sigmoid(cq)).to(X.dtype.element_ty).to(tl.float32)
        k = (ck * tl.sigmoid(ck)).to(X.dtype.element_ty).to(tl.float32)
        v = (cv * tl.sigmoid(cv)).to(X.dtype.element_ty).to(tl.float32)
        q, k = tl.where(kk < DK, q, 0), tl.where(kk < DK, k, 0)
        q *= tl.rsqrt(tl.sum(q*q, 0) + 1e-6) * (DK ** -0.5)
        k *= tl.rsqrt(tl.sum(k*k, 0) + 1e-6)
        decay = tl.exp(tl.load(G + (begin+t)*VH + head).to(tl.float32))
        beta = tl.load(BETA + (begin+t)*VH + head).to(tl.float32)
        state *= decay
        prediction = tl.sum(state * k[:, None], 0)
        delta = (v - prediction) * beta
        state += k[:, None] * delta[None, :]
        result = tl.sum(state * q[:, None], 0)
        destination = tl.load(DESTS + request * MAX_T + t)
        tl.store(SSM + destination * VH * DK * DV + state_offsets, state,
                 (kk[:, None] < DK) & (vv[None, :] < DV))
        tl.store(OUT + (begin+t)*VH*DV + head*DV + vv, result, vv < DV)
        for h in tl.static_range(KERNEL-1):
            relative = t + h - (KERNEL-1) + 1
            old = t + h + 1
            xq = tl.load(X + (begin+relative)*CHANNELS + qch, (relative >= 0) & (kk < DK), 0)
            xk = tl.load(X + (begin+relative)*CHANNELS + kch, (relative >= 0) & (kk < DK), 0)
            xv = tl.load(X + (begin+relative)*CHANNELS + vch, (relative >= 0) & (vv < DV), 0)
            hq = tl.load(CONV + (source*CHANNELS+qch)*(KERNEL-1)+old, (relative < 0) & (kk < DK), 0)
            hk = tl.load(CONV + (source*CHANNELS+kch)*(KERNEL-1)+old, (relative < 0) & (kk < DK), 0)
            hv = tl.load(CONV + (source*CHANNELS+vch)*(KERNEL-1)+old, (relative < 0) & (vv < DV), 0)
            tl.store(CONV + (destination*CHANNELS+qch)*(KERNEL-1)+h, tl.where(relative < 0, hq, xq),
                     (kk < DK) & (head % (VH//KH) == 0))
            tl.store(CONV + (destination*CHANNELS+kch)*(KERNEL-1)+h, tl.where(relative < 0, hk, xk),
                     (kk < DK) & (head % (VH//KH) == 0))
            tl.store(CONV + (destination*CHANNELS+vch)*(KERNEL-1)+h, tl.where(relative < 0, hv, xv), vv < DV)


def fused_verify(layer, mixed, g, beta, bounds, sources, destinations):
    output = mixed.new_empty((mixed.shape[0], layer.value_dim))
    _verify_kernel[(sources.numel(), layer.num_v_heads)](
        mixed, g, beta, layer.conv1d.weight, layer.conv_pool, layer.ssm_pool,
        bounds, sources, destinations, output,
        layer.conv_dim, layer.num_k_heads, layer.num_v_heads, layer.head_k_dim,
        layer.head_v_dim, layer.conv_kernel_size, destinations.shape[1],
        triton.next_power_of_2(layer.head_k_dim), triton.next_power_of_2(layer.head_v_dim),
        num_warps=8 if layer.head_k_dim * layer.head_v_dim >= 4096 else 4,
        enable_fp_fusion=False)
    return output
