class StateMachine:
    def __init__(self):
        self.handlers = {}
        self.startState = None
        self.endStates = []

    def add_state(self, name, handler, end_state=0):
        name = name.upper()
        self.handlers[name] = handler
        if end_state:
            self.endStates.append(name)

    def set_start(self, name):
        self.startState = name.upper()

    def run(self, cargo):
        try:
            handler = self.handlers[self.startState]
        except:
            raise InitializationError("must call .set_start() before .run()")
        if not self.endStates:
            raise InitializationError("at least one state must be an end_state")

        while True:
            (newState, cargo) = handler(cargo)
            if newState.upper() in self.endStates:
                print("reached ", newState)
                break
            else:
                handler = self.handlers[newState.upper()]


positive_adjectives = ["great","super", "fun", "entertaining", "easy"]
negative_adjectives = ["boring", "difficult", "ugly", "bad"]

def start_transitions(txt):
    splitted_txt = txt.split(None,1)
    word, txt = splitted_txt if len(splitted_txt) > 1 else (txt,"")
    if word == "Python":
        newState = "Python_state"
    else:
        newState = "error_state"
    return (newState, txt)

def python_state_transitions(txt):
    splitted_txt = txt.split(None,1)
    word, txt = splitted_txt if len(splitted_txt) > 1 else (txt,"")
    if word == "is":
        newState = "is_state"
    else:
        newState = "error_state"
    return (newState, txt)

def is_state_transitions(txt):
    splitted_txt = txt.split(None,1)
    word, txt = splitted_txt if len(splitted_txt) > 1 else (txt,"")
    if word == "not":
        newState = "not_state"
    elif word in positive_adjectives:
        newState = "pos_state"
    elif word in negative_adjectives:
        newState = "neg_state"
    else:
        newState = "error_state"
    return (newState, txt)

def not_state_transitions(txt):
    splitted_txt = txt.split(None,1)
    word, txt = splitted_txt if len(splitted_txt) > 1 else (txt,"")
    if word in positive_adjectives:
        newState = "neg_state"
    elif word in negative_adjectives:
        newState = "pos_state"
    else:
        newState = "error_state"
    return (newState, txt)

def neg_state(txt):
    print("Hallo")
    return ("neg_state", "")


def start(txt, fun):
    return ("first", txt, "", fun)

def pair(txt, seq, fun):
    if txt:
        item = txt.pop()
        if fun(item):
            txt.join(item)
            return ("buffer", txt, seq, fun)
        else:
            seq.join(item)
    return ("end", seq)

def second(txt, seq, fun):
    temp = ""
    while txt:
        item = txt.pop()
        if fun(item):
            temp.join(item)
            if len(temp) > 3:
                seq.join(temp)
        else:
            return ("first", txt, seq, fun)
    return ("end", seq)

def f(a):
    if a == 'B':
        return True
    else:
        return False

try_m = StateMachine()
try_m.add_state("start", start)
try_m.add_state("first", first)
try_m.add_state("second", second)
try_m.add_state("end", None, end_state=1)
try_m.set_start("start")
try_m.run(("AAABBBAA0B", f))

# m = StateMachine()
# m.add_state("Start", start_transitions)
# m.add_state("Python_state", python_state_transitions)
# m.add_state("is_state", is_state_transitions)
# m.add_state("not_state", not_state_transitions)
# m.add_state("neg_state", None, end_state=1)
# m.add_state("pos_state", None, end_state=1)
# m.add_state("error_state", None, end_state=1)
# m.set_start("Start")
# m.run("Python is great")
# m.run("Python is difficult")
# m.run("Perl is ugly")