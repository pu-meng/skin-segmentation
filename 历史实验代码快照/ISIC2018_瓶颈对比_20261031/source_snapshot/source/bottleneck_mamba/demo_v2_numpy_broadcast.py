"""第二版第5、6、7步的 NumPy 演示；数值为演示构造，不读取模型。"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# B：batch，L：位置数，D：通道数，N：状态分量数。
# 沿用真实代码排列：T、delta 是 [B,D,L]；B_coef、C 是 [B,N,L]。
# B_coef 是公式里的 B；d_skip 是公式里的 D，避免与维度名字重名。
B, L, D, N = 2, 2, 2, 3


def show(name, x):
    print(f"\n{name}，shape = {x.shape}")
    print(x)


# 每个二维小数组是一张图：行是 D，列是 L。
T = np.array([
    [[2., 4.], [3., 6.]],       # b=0
    [[5., 8.], [7., 10.]],      # b=1
])
delta = np.array([
    [[np.log(2), np.log(2)], [np.log(2), np.log(2)/2]],
    [[np.log(2), np.log(4)], [np.log(2), np.log(4)/2]],
])

# 每个二维小数组是一张图：行是 N，列是 L。
B_coef = np.array([
    [[1., 2.], [2., 3.], [3., 4.]],
    [[2., 1.], [1., 3.], [4., 2.]],
])
C = np.array([
    [[1., 1.], [1., 2.], [1., -1.]],
    [[1., 2.], [1., -1.], [1., 1.]],
])

# A_log、d_skip 在模型中是可学习参数；这里直接指定演示值。
# A 的行是 D，列是 N；不含 batch 和位置轴。
A_log = np.log(np.array([[1., 2., 3.], [2., 4., 6.]]))
A = -np.exp(A_log)
d_skip = np.array([0.5, -2.])

show('T [B,D,L]', T)
show('delta [B,D,L]', delta)
show('A [D,N]', A)
show('B_coef [B,N,L]', B_coef)
show('C [B,N,L]', C)

# state 不含 L：它只保存走到当前时刻的状态。
# 每张图一块 [D,N]，每个通道有 N 个状态数。
state = np.zeros((B, D, N))
outputs = []

for t in range(L):
    print(f"\n{'=' * 24} t = {t} {'=' * 24}")

    # 固定 t 后，从三维数组取出二维数组。
    delta_t = delta[:, :, t]     # [B,D]
    T_t = T[:, :, t]             # [B,D]
    B_t = B_coef[:, :, t]        # [B,N]
    C_t = C[:, :, t]             # [B,N]
    h_prev = state.copy()        # [B,D,N]

    show('delta_t', delta_t)
    show('h_prev', h_prev)

    # 第5步：delta 和 A 相乘。
    # None 只插入一条长度为1的轴，不改变原有数值。
    # [B,D,1] * [1,D,N] -> [B,D,N]
    delta_3d = delta_t[:, :, None]
    A_3d = A[None, :, :]
    show('delta_t[:, :, None]', delta_3d)
    show('A[None, :, :]', A_3d)

    delta_A = delta_3d * A_3d
    show('delta_A = delta_3d * A_3d', delta_A)

    E = np.exp(delta_A)
    show('E = exp(delta_A)', E)

    # E 和 h_prev 完全同形状，逐格相乘，没有求和。
    # [B,D,N] * [B,D,N] -> [B,D,N]
    E_h_prev = E * h_prev
    show('E * h_prev', E_h_prev)

    # 第二版的写入项：不乘 delta。
    # [B,D,1] * [B,1,N] -> [B,D,N]
    T_3d = T_t[:, :, None]
    B_3d = B_t[:, None, :]
    show('T_t[:, :, None]', T_3d)
    show('B_t[:, None, :]', B_3d)

    J = T_3d * B_3d
    show('J = T_3d * B_3d', J)

    h_t = E_h_prev + J
    show('h_t = E * h_prev + J', h_t)

    # 把全部 h_t[b,d,n] 展开成标量计算，核对数组里的每一格。
    print('\n逐个计算 h_t[b,d,n]：')
    for b in range(B):
        for d in range(D):
            for n in range(N):
                value = (
                    np.exp(delta_t[b, d] * A[d, n]) * h_prev[b, d, n]
                    + T_t[b, d] * B_t[b, n]
                )
                print(
                    f'h_{t}[{b},{d},{n}] = '
                    f'exp({delta_t[b,d]:.4f} * {A[d,n]:g}) '
                    f'* {h_prev[b,d,n]:g} '
                    f'+ {T_t[b,d]:g} * {B_t[b,n]:g} '
                    f'= {value:.4f}'
                )
                assert np.isclose(value, h_t[b, d, n])

    # 第6步：C 只沿 D 轴广播；每个通道共用同一组 C。
    # [B,D,N] * [B,1,N] -> [B,D,N]
    C_3d = C_t[:, None, :]
    show('C_t[:, None, :]', C_3d)
    C_h = h_t * C_3d
    show('C_h = h_t * C_3d', C_h)

    # 只对最后一条 N 轴求和；D 轴仍然保留。
    # [B,D,N] -> [B,D]
    Y0_t = C_h.sum(axis=-1)
    show('Y0_t = C_h.sum(axis=-1)', Y0_t)
    for b in range(B):
        for d in range(D):
            terms = ' + '.join(
                f'({h_t[b,d,n]:g} * {C_t[b,n]:g})' for n in range(N)
            )
            print(f'Y0[{b},{d},{t}] = {terms} = {Y0_t[b,d]:g}')

    outputs.append(Y0_t)
    state = h_t

# 每个位置的 Y0_t 都是 [B,D]，按最后一条轴叠成 [B,D,L]。
Y0 = np.stack(outputs, axis=-1)
show('Y0 [B,D,L]', Y0)

# 第7步：逐通道直连。
# [1,D,1] * [B,D,L] -> [B,D,L]
# d=0 的所有图片、所有位置乘0.5；d=1 的全部乘-2。
show('d_skip [D]', d_skip)
D_3d = d_skip[None, :, None]
show('d_skip[None, :, None]', D_3d)
D_T = D_3d * T
show('D_T = D_3d * T', D_T)
Y1 = Y0 + D_T
show('Y1 = Y0 + D_T', Y1)

print('\n逐个计算 Y1[b,d,t]：')
for b in range(B):
    for d in range(D):
        for t in range(L):
            value = Y0[b, d, t] + d_skip[d] * T[b, d, t]
            print(
                f'Y1[{b},{d},{t}] = {Y0[b,d,t]:g} '
                f'+ {d_skip[d]:g} * {T[b,d,t]:g} = {value:g}'
            )
            assert np.isclose(value, Y1[b, d, t])

# 额外核对：以上分步广播与真实代码的 einsum 写法一致。
transitions = np.exp(np.einsum('bdt,dn->bdtn', delta, A))
injections = np.einsum('bdt,bnt->bdtn', T, B_coef)
reference_state = np.zeros((B, D, N))
reference_outputs = []
for t in range(L):
    reference_state = (
        transitions[:, :, t, :] * reference_state
        + injections[:, :, t, :]
    )
    reference_outputs.append(
        (reference_state * C[:, None, :, t]).sum(axis=-1)
    )
np.testing.assert_allclose(Y0, np.stack(reference_outputs, axis=-1))
print('\n检查通过：逐格计算、分步广播和真实代码的 einsum 写法结果一致。')
