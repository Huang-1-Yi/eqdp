"""
典型应用场景：
• 数据验证（如检查所有值是否为正数、是否为特定类型）。
• 嵌套结构的数据处理（如求和、求最大值）。
• 条件检查（如配置文件中所有参数是否在合法范围内）。

这段代码提供了三个函数来处理嵌套字典，主要用于对嵌套结构中的叶子节点进行映射、归约和条件检查操作。具体功能如下：
1. `nested_dict_map(f, x)`:
   • 功能：
   递归遍历嵌套字典 `x`,对每个叶子节点（非字典的值）应用函数 `f`，生成一个结构相同但所有叶子值被 `f` 处理过的新字典。
   • 示例：
     # 输入：{'a': 1, 'b': {'c': 2}}
     # 调用：nested_dict_map(lambda x: x*2, x)
     # 输出：{'a': 2, 'b': {'c': 4}}

2. `nested_dict_reduce(f, x)`:
   • 功能：
   递归归约嵌套字典 `x`。对于每个层级的子节点，若为字典则递归归约其值，否则直接取值；最终通过归约函数 `f` 将所有结果合并为一个值。

   • 示例：
     # 输入：{'a': 1, 'b': {'c': 2, 'd': 3}}
     # 调用：nested_dict_reduce(lambda x,y: x+y, x)
     # 计算逻辑：1 + (2+3) = 6
     # 输出：6

3. `nested_dict_check(f, x)`:
   • 功能：
   检查嵌套字典所有叶子节点是否满足条件 `f`。
   先用 `nested_dict_map` 生成布尔字典，再用 `nested_dict_reduce` 通过逻辑与 (`and`) 归约，最终返回 `True`（所有叶子满足条件）或 `False`。

   • 示例：
     # 输入：{'a': 5, 'b': {'c': -3, 'd': 10}}
     # 调用：nested_dict_check(lambda x: x>0, x)
     # 过程：生成 {'a': True, 'b': {'c': False, 'd': True}}，归约为 True and False → False
     # 输出：False
"""
import functools

def nested_dict_map(f, x):
    """
    Map f over all leaf of nested dict x
    """

    if not isinstance(x, dict):
        return f(x)
    y = dict()
    for key, value in x.items():
        y[key] = nested_dict_map(f, value)
    return y

def nested_dict_reduce(f, x):
    """
    Map f over all values of nested dict x, and reduce to a single value
    """
    if not isinstance(x, dict):
        return x

    reduced_values = list()
    for value in x.values():
        reduced_values.append(nested_dict_reduce(f, value))
    y = functools.reduce(f, reduced_values)
    return y


def nested_dict_check(f, x):
    bool_dict = nested_dict_map(f, x)
    result = nested_dict_reduce(lambda x, y: x and y, bool_dict)
    return result
