import re
from typing import Optional, Dict, Tuple, Union
from goal_elem import *
    

class GoalFileParser(object):
    """
    An object that parses a given goal file.

    The EBNF of the goal file is as follows:
    <goal-file> ::= <global-ranks> <whitespace> <rank-ops>+
    <global-ranks> ::= "num_ranks" <whitespace> <digits>
    <whitespace> ::= ( " " | "\t" | "\n" )+
    <digits> ::= <digit>+
    <digit> ::= "0" | "1" | "2" | "3" | "4" |
                "5" | "6" | "7" | "8" | "9"
    <rank-ops> ::= "rank" <whitespace> <digits> <whitespace> 
                   "{" ( (<local-op> | <dependency>) <whitespace> )+ "}"
    <local-op> ::= <send-op> | <recv-op> | <calc-op> | <dependency>
    # example: send 1 b to 2 tag 3
    <send-op> ::= <label> ":" <whitespace> "send" <whitespace> <digits> "b" 
                  <whitespace> "to" <whitespace> <digits> (<whitespace> "tag" <whitespace> <digits>)?
    # example: recv 1 b from 2 tag 3
    <recv-op> ::= <label> ":" <whitespace> "recv" <whitespace> <digits> "b"
                  <whitespace> "from" <whitespace> <digits> (<whitespace> "tag" <whitespace> <digits>)?
    # example: calc 1: 2
    <calc-op> ::= <label> ":" <whitespace> "calc" <whitespace> <digits>
    <label> ::= "l" <digits>
    # example: l1 require l2
    <dependency> ::= <label> <whitespace> <req> <whitespace> <label>
    <req> ::= "irequires" | "requires"


    TODO: The parser is currently implemented purely
          using regex. This is not a good idea. We
          should use a proper parser generator like
          pyparsing or PLY.
    """
    def __init__(self) -> None:
        """
        Initialize the goal file parser.
        """
        pass

    @staticmethod
    def __parse_meta_value(raw: str) -> Union[int, str]:
        """
        Parse a metadata value into int when possible.
        """
        try:
            return int(raw)
        except ValueError:
            return raw

    def __parse_metadata(self, trailing_tokens: str) -> Dict[str, Union[int, str]]:
        """
        Parses optional key-value metadata that can trail an op line, e.g.
        `cpu 3 nic 0 stream 1` or `stream=1`.
        """
        trailing_tokens = trailing_tokens.strip()
        if len(trailing_tokens) == 0:
            return {}

        metadata: Dict[str, Union[int, str]] = {}
        tokens = trailing_tokens.split()
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if "=" in token:
                key, value = token.split("=", 1)
                if key:
                    metadata[key] = self.__parse_meta_value(value)
                i += 1
                continue

            if i + 1 < len(tokens):
                key = token
                value = tokens[i + 1]
                # Skip malformed pair like "key value=1", the next loop
                # iteration will handle the key=value token.
                if "=" not in value:
                    metadata[key] = self.__parse_meta_value(value)
                    i += 2
                    continue

            i += 1

    def parse_line(self, line: str) -> Optional[GoalElement]:
        """
        Parses a single given line in the goal file and converts
        it into a GoalElement object with the help
        of regex, which can then by used by the dependency graph generator
        to generate the dependency graph.

        @param line: The line to be parsed.
        @return: The parsed goal element. If the line is empty,
        then None is returned.
        """
        # Skips the line if it is empty
        if len(line.strip()) == 0:
            return None

        # Matches the global ranks
        match = re.match(r"^num_ranks\s+(\d+)$", line)
        if match:
            return GlobalRanks(int(match.group(1)))
        # Matches the rank start
        match = re.match(r"^rank\s+(\d+)\s+{$", line)
        if match:
            return RankStart(int(match.group(1)))
        # Matches the rank end
        match = re.match(r"^}$", line)
        if match:
            return RankEnd()
        # Matches the send operation
        match = re.match(r"^l(\d+)\s*:\s*send\s+(\d+)b\s+to\s+(\d+)(?:\s+tag\s+(\d+))?(.*)$", line)
        if match:
            tag = int(match.group(4)) if match.group(4) is not None else None
            metadata = self.__parse_metadata(match.group(5))
            return SendOp(int(match.group(1)), int(match.group(2)),
                          int(match.group(3)), tag, metadata)
        # Matches the recv operation
        match = re.match(r"^l(\d+)\s*:\s*recv\s+(\d+)b\s+from\s+(\d+)(?:\s+tag\s+(\d+))?(.*)$", line)
        if match:
            tag = int(match.group(4)) if match.group(4) is not None else None
            metadata = self.__parse_metadata(match.group(5))
            return RecvOp(int(match.group(1)), int(match.group(2)),
                          int(match.group(3)), tag, metadata)
        # Matches the calc operation
        match = re.match(r"^l(\d+)\s*:\s*calc\s+(\d+)(.*)$", line)
        if match:
            metadata = self.__parse_metadata(match.group(3))
            return CalcOp(int(match.group(1)), int(match.group(2)), metadata)
        # Matches collective marker/event operation
        match = re.match(r"^l(\d+)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s+(\d+)\s+bytes(.*)$", line)
        if match:
            metadata = self.__parse_metadata(match.group(4))
            return CollectiveOp(
                int(match.group(1)),
                match.group(2),
                int(match.group(3)),
                metadata,
            )
        # Matches the dependency
        match = re.match(r"^l(\d+)\s+(irequires|requires)\s+l(\d+)$", line)
        if match:
            # Checks if the dependency is an irequire or a require
            is_irequire = match.group(2) == "irequires"
            return Dependency(int(match.group(3)),
                              int(match.group(1)), is_irequire)
        # If none of the above matches, then the line is invalid
        raise ValueError("Invalid line in the goal file: {}".format(line))



class CommDepFileParser(object):
    """
    An object that parses a given communication dependency file.
    Each line in the communication dependency files has the following format:
    <src-rank>,<src-op-label>,<dst-rank>,<dst-op-label>
    """
    def __init__(self) -> None:
        """
        Initialize the communication dependency file parser.
        """
        pass

    def parse_line(self, line: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Parses a single given line in the communication dependency file
        and converts it into a tuple of tuples of ints, which can then
        by used by the dependency graph generator to generate the
        dependency graph.

        @param line: The line to be parsed.
        @return: A tuple of tuples of ints, where the first tuple
                 represents the source vertex and the second tuple
                 represents the destination vertex.
        """
        tokens = line.split(",")
        assert len(tokens) == 4
        src_rank, src_op_label, dst_rank, dst_op_label = tokens
        
        # Convert the tokens to ints
        src_rank = int(src_rank)
        src_op_label = int(src_op_label)
        dst_rank = int(dst_rank)
        dst_op_label = int(dst_op_label)

        return ((src_rank, src_op_label),(dst_rank, dst_op_label))
