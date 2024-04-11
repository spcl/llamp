from typing import Optional, List, Union, Tuple


class GoalElement(object):
    """
    An object that represents a single element in the goal file,
    which should be inherited by all the child classes.
    """
    def __init__(self) -> None:
        pass


class GlobalRanks(GoalElement):
    """
    An object that represents the global ranks in the goal file.
    This element should be the first line in the goal file.
    """
    def __init__(self, num_ranks: int) -> None:
        """
        Initialize the global ranks object.
        @param num_ranks: The number of ranks in the MPI program.
        """
        self.num_ranks = num_ranks


class RankStart(object):
    """
    An object that represents the start of a rank in the goal file.
    """
    def __init__(self, rank: int) -> None:
        """
        Initialize the rank start object.
        @param rank: The rank number.
        """
        self.rank = rank
    

class RankEnd(object):
    """
    An object that represents the end of a rank in the goal file.
    """
    def __init__(self) -> None:
        pass


class SendOp(GoalElement):
    """
    An object that represents a send operation in the goal file.
    """
    def __init__(self, label: int, data_size: int,
                 dst: int, tag: Optional[int] = None) -> None:
        """
        Initialize the send operation object.
        @param label: The label of the send operation.
        @param data_size: The size of the data being sent.
        @param dst: The destination rank of the send operation.
        @param tag: The tag of the send operation.
        """
        self.label = label
        self.data_size = data_size
        self.dst = dst
        self.tag = tag


class RecvOp(GoalElement):
    """
    An object that represents a recv operation in the goal file.
    """
    def __init__(self, label: int, data_size: int,
                 src: int, tag: int) -> None:
        """
        Initialize the recv operation object.
        @param label: The label of the recv operation.
        @param data_size: The size of the data being received.
        @param src: The source rank of the recv operation.
        @param tag: The tag of the recv operation.
        """
        self.label = label
        self.data_size = data_size
        self.src = src
        self.tag = tag


class CalcOp(GoalElement):
    """
    An object that represents a calc operation in the goal file.
    """
    def __init__(self, label: int, cost: int) -> None:
        """
        Initialize the calc operation object.
        @param label: The label of the calc operation.
        @param cost: The computation cost of the calc operation.
        """
        self.label = label
        self.cost = cost


class Dependency(GoalElement):
    """
    An object that represents a local dependency in the goal file.
    """
    def __init__(self, src_label: int, dst_label: int, 
                 is_irequire: bool) -> None:
        """
        Initialize the dependency object.
        @param src: The source label of the dependency.
        @param dst: The destination label of the dependency.
        @param is_irequire: Whether the dependency is an irequire dependency.
        """
        self.src_label = src_label
        self.dst_label = dst_label
        self.is_irequire = is_irequire
