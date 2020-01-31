def _c(char):
    def append_char(text):
        return text + char
    return append_char

class _DotData:
    def __init__(self, label, next_state, action):
        self.label = label
        self.next_state = next_state
        self.action = action

def _build_noninitial_state(state_name, options):
    state = [_DotData("", state_name, _c("")) for _ in range(9)]
    state[4] = _DotData("Go back", "0", _c(""))
    input_dot_indexes = [0, 2, 6, 8]
    for i in range(len(options)):
        option = options[i]
        if option == "Space":
            action = _c(" ")
        elif option == "Backspace":
            action = lambda text: text[:-1]
        else:
            action = _c(option)
        state[input_dot_indexes[i]] = _DotData(option, "0", action)
    return state

class TextInputAutomata:
    STATES = {
            "0": [_DotData("Backspace Space", "1", _c("")),
                  _DotData("A B C", "2", _c("")),
                  _DotData("D E F", "3", _c("")),
                  _DotData("G H I", "4", _c("")),
                  _DotData("J K L", "5", _c("")),
                  _DotData("M N O", "6", _c("")),
                  _DotData("P Q R S", "7", _c("")),
                  _DotData("T U V", "8", _c("")),
                  _DotData("W X Y Z", "9", _c(""))],
            "1": _build_noninitial_state("1", ["Backspace", "Space"]),
            "2": _build_noninitial_state("2", ["A", "B", "C"]),
            "3": _build_noninitial_state("3", ["D", "E", "F"]),
            "4": _build_noninitial_state("4", ["G", "H", "I"]),
            "5": _build_noninitial_state("5", ["J", "K", "L"]),
            "6": _build_noninitial_state("6", ["M", "N", "O"]),
            "7": _build_noninitial_state("7", ["P", "Q", "R", "S"]),
            "8": _build_noninitial_state("8", ["T", "U", "V"]),
            "9": _build_noninitial_state("9", ["W", "X", "Y", "Z"])
    }

    def __init__(self):
        self.state_name = "0"

    def labels(self):
        return list(map(lambda dot_data: dot_data.label,
                TextInputAutomata.STATES[self.state_name]))

    def transition(self, dot_index, text):
        dot_data = TextInputAutomata.STATES[self.state_name][dot_index]
        self.state_name = dot_data.next_state
        return dot_data.action(text)
