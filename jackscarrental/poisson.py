from scipy.stats import poisson
import numpy as np

class Poisson(object):
    """
    这个类用于计算泊松分布的概率质量函数(PMF)和累积分布函数(SF)。
    它使用缓存机制来优化性能，避免重复计算相同参数的分布。
    """
    # 缓存泊松分布的 PMF 和 SF 值
    cache_pmf = {}
    cache_sf = {}
    # 缓存计算结果
    cache = {}
    # 定义最大截断值
    MAX_CUTOFF = 25

    @classmethod
    def pmf_series(cls, mu, cutoff):
        """
        获取泊松分布的 PMF 序列。

        :param mu: 泊松分布的平均值（必须是整数）。
        :param cutoff: 计算 PMF 的截断值（必须是整数）。
        :return: 返回截断至 cutoff 的 PMF 数组。
        """
        assert isinstance(mu, int), "mu 应为整数。"
        assert isinstance(cutoff, int), "cutoff 应为整数。"

        # 如果缓存中没有，则计算 PMF 序列
        if (mu, cutoff) not in cls.cache:
            cls._calculate_pmf_series(mu, cutoff)

        return cls.cache[(mu, cutoff)]

    @classmethod
    def _calculate_pmf_series(cls, mu, cutoff):
        """
        计算并缓存泊松分布的 PMF 序列。

        :param mu: 泊松分布的平均值。
        :param cutoff: 计算 PMF 的截断值。
        """
        # 如果 mu 不在缓存中，则计算并缓存 PMF 和 SF
        if mu not in cls.cache_pmf:
            print(f"计算 mu={mu} 的泊松分布...")
            cls.cache_pmf[mu] = poisson.pmf(np.arange(cls.MAX_CUTOFF + 1), mu)
            cls.cache_sf[mu] = poisson.sf(np.arange(cls.MAX_CUTOFF + 1), mu)

        # 复制 PMF 数组，并将最后一个元素增加 SF 的值
        out = np.copy(cls.cache_pmf[mu][:cutoff+1])
        out[-1] += cls.cache_sf[mu][cutoff]

        cls.cache[(mu, cutoff)] = out


if __name__ == '__main__':
    pass

