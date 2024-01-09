import numpy as np
from .poisson import Poisson

class PolicyIterationSolver(object):
    # 定义基本参数
    capacity = 20  # 最大容量
    rental_reward = 10.  # 出租奖励
    moving_cost = 2.  # 移动成本
    max_moving = 5  # 最大移动数

    # 错误行动成本，设定为非负数。
    # 当 bad_action_cost = 0 时，错误行动不受惩罚，否则根据设定值进行惩罚。
    bad_action_cost = 0.1

    # 平均请求与返回数量
    request_mean_G1 = 3
    request_mean_G2 = 4
    return_mean_G1 = 3
    return_mean_G2 = 2

    # 折扣因子
    discount = 0.09

    # 策略评估误差
    PolicyEvaluationError = 0.01

    # 初始化策略和价值
    def __init__(self):
        self.policy = np.zeros([self.capacity + 1]*2, int)
        self.value = np.zeros([self.capacity + 1]*2)

        self._reward1 = self.expected_rental_reward(self.request_mean_G1)
        self._reward2 = self.expected_rental_reward(self.request_mean_G2)

        assert self.bad_action_cost >= 0

    # 贝尔曼方程
    def bellman(self, action, s1, s2):
        transp1 = self.transition_probability(s1, self.request_mean_G1, self.return_mean_G1, -action)
        transp2 = self.transition_probability(s2, self.request_mean_G2, self.return_mean_G2, action)
        transp = np.outer(transp1, transp2)

        return self._reward1[s1] + self._reward2[s2] - self.expected_moving_cost(s1, s2, action) + \
               self.discount * sum((transp * self.value).flat)

    # 策略评估
    def policy_evaluation(self):
        ''' 保持策略固定并更新价值。 '''
        while True:
            diff = 0.
            it = np.nditer([self.policy], flags=['multi_index'])

            while not it.finished:
                action = it[0]
                s1, s2 = it.multi_index

                _temp = self.value[s1, s2]

                self.value[s1, s2] = self.bellman(action=action, s1=s1, s2=s2)

                diff = max(diff, abs(self.value[s1, s2] - _temp))

                it.iternext()

            #print(diff)
            if diff < self.PolicyEvaluationError:
                break

    # 策略更新
    def policy_update(self):
        is_policy_changed = False

        it = np.nditer([self.policy], flags=['multi_index'])
        while not it.finished:
            s1, s2 = it.multi_index

            _max_val = -1
            _pol = None

            for act in range(-self.max_moving, self.max_moving + 1):
                _val = self.bellman(action=act, s1=s1, s2=s2)
                if _val > _max_val:
                    _max_val = _val
                    _pol = act

            if self.policy[s1, s2] != _pol:
                is_policy_changed = True
                self.policy[s1, s2] = _pol

            it.iternext()

        return is_policy_changed

    # 计算预期移动成本
    def expected_moving_cost(self, s1, s2, action):
        if action == 0:
            return 0.

        # 从状态 s1 移动到状态 s2
        if action > 0:
            p = self.transition_probability(s1, self.request_mean_G1, self.return_mean_G1)
            cost = self._gen_move_cost_array(action)
            return cost.dot(p)

        # 从状态 s2 移动到状态 s1
        p = self.transition_probability(s2, self.request_mean_G2, self.return_mean_G2)
        cost = self._gen_move_cost_array(action)
        return cost.dot(p)

    # 生成移动成本数组
    def _gen_move_cost_array(self, action):
        '''
         根据移动数生成成本数组。

         当可移动车辆少于 action 时，视为错误行动。

         当 self.bad_move_cost == 0 时，错误行动不受惩罚，系统将尽可能多地移动车辆。

         当 self.bad_move_cost > 0 时，错误行动会根据这个变量的值受到惩罚。

         :param action: 将从车库 1 移动到车库 2 的车辆数。
         :return:
        '''
        _action = abs(action)

        # 不惩罚错误行动
        if self.bad_action_cost == 0:
            cost = np.asarray(
                [ii if ii < _action else _action for ii in range(self.capacity+1)]
            ) * self.moving_cost

        # 惩罚错误行动
        else:
            cost = np.asarray(
                [self.bad_action_cost if ii < _action else _action for ii in range(self.capacity + 1)]
            ) * self.moving_cost
        return cost

    # 计算预期租赁奖励
    @classmethod
    def expected_rental_reward(cls, expected_request):
        return np.asarray([cls._state_reward(s, expected_request) for s in range(cls.capacity + 1)])

    # 计算单个状态的奖励
    @classmethod
    def _state_reward(cls, s, mu):
        rewards = cls.rental_reward * np.arange(s + 1)
        p = Poisson.pmf_series(mu, cutoff=s)
        return rewards.dot(p)

    # 计算转移概率
    def transition_probability(self, s, req, ret, action=0):
        '''
         计算转移概率。

         :param s: 当前状态
         :param req: 请求的平均值
         :param ret: 返回的平均值
         :param action: 行动。正数表示移入，负数表示移出。
         :return: 转移概率。
        '''

        _ret_sz = self.max_moving + self.capacity

        p_req = Poisson.pmf_series(req, s)
        p_ret = Poisson.pmf_series(ret, _ret_sz)
        p = np.outer(p_req, p_ret)

        transp = np.asarray([p.trace(offset) for offset in range(-s, _ret_sz + 1)])

        assert abs(action) <= self.max_moving, "行动不能大于 %s." % self.max_moving

        # 没有移动车辆
        if action == 0:
            transp[20] += sum(transp[21:])
            return transp[:21]

        # 从车库 1 移动到车库 2
        if action > 0:
            transp[self.capacity-action] += sum(transp[self.capacity-action+1:])
            transp[self.capacity-action+1:] = 0

            return np.roll(transp, shift=action)[:self.capacity+1]

        # 从车库 2 移动到车库 1
        action = -action
        transp[action] += sum(transp[:action])
        transp[:action] = 0

        transp[action+self.capacity] += sum(transp[action+self.capacity+1:])
        transp[action+self.capacity+1:] = 0

        return np.roll(transp, shift=-action)[:self.capacity+1]

    # 策略迭代
    def policy_iteration(self):
        '''
         注意：策略不断在两个或更多同样优秀的策略之间切换的情况尚未考虑。
         :return:
        '''
        self.policy_evaluation()
        while self.policy_update():
            self.policy_evaluation()


# 主函数
if __name__ == '__main__':
    solver = PolicyIterationSolver()

    for ii in range(4):
        solver.policy_evaluation()
        solver.policy_update()

    print(solver.policy)

    import matplotlib.pylab as plt

    plt.subplot(121)
    CS = plt.contour(solver.policy, levels=range(-6, 6))
    plt.clabel(CS)
    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.axis('equal')
    plt.xticks(range(21))
    plt.yticks(range(21))
    plt.grid('on')

    plt.subplot(122)
    plt.pcolor(solver.value)
    plt.colorbar()
    plt.axis('equal')

    plt.show()
