from transitions import Machine, State

class Matter(object):
    def say_hello(self): print("hello, new state!")
    def say_goodbye(self): print("goodbye, old state!")
    def on_enter_A(self): print("We've just entered state A!")

lump = Matter()

# The states
states=['solid', 'liquid', 'gas', 'plasma']

# And some transitions between states. We're lazy, so we'll leave out
# the inverse phase transitions (freezing, condensation, etc.).
transitions = [
    { 'trigger': 'melt', 'source': 'solid', 'dest': 'liquid' },
    { 'trigger': 'evaporate', 'source': 'liquid', 'dest': 'gas' },
    { 'trigger': 'sublimate', 'source': 'solid', 'dest': 'gas' },
    { 'trigger': 'ionize', 'source': 'gas', 'dest': 'plasma' }
]
# Initialize
machine = Machine(lump, states=states, transitions=transitions, initial='liquid')
# Now lump maintains state...
print(lump.state)
lump.evaporate()
print(lump.state)
lump.trigger('ionize')
print(lump.state)


# Create a list of 3 states to pass to the Machine
# initializer. We can mix types; in this case, we
# pass one State, one string, and one dict.
# Same states as above, but now we give StateA an exit callback
states = [
    State(name='solid', on_exit=['say_goodbye']),
    'liquid',
    { 'name': 'gas' }
    ]
lump = Matter()
machine = Machine(lump, states=['A', 'B', 'C'])


class Parsing(Machine):

    def __init__(self, txt=""):
        # check parameters
        if not isinstance(txt, str):
            raise TypeError
        # states machine
        states = [
            State(name='start'),
            State(name='pair',      on_enter="flow_sequence"),
            State(name='buffer',    on_enter="check_b"),
            State(name='write',     on_enter="fun_write"),
        ]
        transitions = [
            {'trigger': 'run',      'source':'start',   'dest': 'pair'},
            {'trigger': 'not_b',    'source':'pair',    'dest': 'pair'},
            {'trigger': 'find_b',   'source':'pair',    'dest': 'buffer'},
            {'trigger': 'not_b',    'source':'buffer',  'dest': 'pair'},
            {'trigger': 'find_b',   'source':'buffer',  'dest': 'buffer'},
            {'trigger': 'write_b',    'source':'buffer',  'dest': 'write'},
            {'trigger': 'not_b',    'source':'write',   'dest': 'pair'}
        ]
        self.machine = Machine.__init__(self, states=states, transitions=transitions, initial='start')
        # parameters
        self.txt = list(txt)
        self.seq = ""

    def flow_sequence(self):
        if self.txt and Parsing.fun(self.txt[0]):# find b
            self.find_b()
        elif self.txt:# not find b
            self.seq=self.seq+self.txt.pop(0)
            self.not_b()
    def check_b(self):
        tmp = ""
        if self.txt:
            item = self.txt[0]
            if not Parsing.fun(item):
                self.write_b(tmp)
            tmp.join(self.txt.pop(0))
            self.check_b()
        else:
            self.write_b(tmp)
    def fun_write(self, tmp):
        if Parsing.fun2(len(tmp)):
            for item in tmp: self.seq.join(item)
        self.not_b()

    # static methods
    @staticmethod
    def fun(item):
        if item == 'B':
            return True
        return False
    @staticmethod
    def fun2(item):
        if item >= 3:
            return True
        return False


lump = Parsing(txt="abB")
lump.run()
print(lump.seq)
# def start(txt, fun):
#     return ("first", txt, "", fun)
#
# def first(txt, seq, fun):
#     while txt:
#         item = txt.pop()
#         if fun(item):
#             txt.join(item)
#             return ("second", txt, seq, fun)
#         else:
#             seq.join(item)
#     return ("end", seq)
#
# def second(txt, seq, fun):
#     temp = ""
#     while txt:
#         item = txt.pop()
#         if fun(item):
#             temp.join(item)
#             if len(temp) > 3:
#                 seq.join(temp)
#         else:
#             return ("first", txt, seq, fun)
#     return ("end", seq)
#
# def f(a):
#     if a == 'B':
#         return True
#     else:
#         return False
#
# try_m = StateMachine()
# try_m.add_state("start", start)
# try_m.add_state("first", first)
# try_m.add_state("second", second)
# try_m.add_state("end", None, end_state=1)
# try_m.set_start("start")
# try_m.run(("AAABBBAA0B", f))