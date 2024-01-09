from jackscarrental import PolicyIterationSolver
solver = PolicyIterationSolver()

# 进行策略迭代
for ii in range(4):
    solver.policy_evaluation()
    solver.policy_update()

# 输出策略矩阵
print(solver.policy)

# 设置matplotlib绘图
import matplotlib.pylab as plt

# 设置画布大小
plt.figure(figsize=(14, 6))

# 绘制策略图
plt.subplot(121)
CS = plt.contour(solver.policy, levels=range(-6, 7), cmap='viridis')  # 使用viridis颜色图
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Optimal Policy (Action per State)')
plt.xlabel('Number of Cars at Location 1')
plt.ylabel('Number of Cars at Location 2')
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.axis('equal')
plt.xticks(range(0, 21, 2))  # 仅显示0至20的偶数刻度
plt.yticks(range(0, 21, 2))
plt.grid(True)

# 绘制价值函数图
plt.subplot(122)
value_plot = plt.pcolor(solver.value, cmap='hot')  # 使用hot颜色图
plt.colorbar(value_plot)
plt.title('Value Function')
plt.xlabel('Number of Cars at Location 1')
plt.ylabel('Number of Cars at Location 2')
plt.axis('equal')
plt.xticks(range(0, 21, 2))
plt.yticks(range(0, 21, 2))

# 调整布局
plt.tight_layout()
plt.show()


