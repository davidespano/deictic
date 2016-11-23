from pomegranate import *

random.seed(0)
state1 = State( UniformDistribution(0.0, 1.0), name="uniform" )
state2 = State( NormalDistribution(0, 2), name="normal" )

model = HiddenMarkovModel( name="ExampleModel" )
model.add_state( state1 )
model.add_state( state2 )

model.add_transition( model.start, state1, 0.5 )
model.add_transition( model.start, state2, 0.5 )

model.add_transition( state1, state1, 0.4 )
model.add_transition( state1, state2, 0.4 )
model.add_transition( state2, state2, 0.4 )
model.add_transition( state2, state1, 0.4 )

model.add_transition( state1, model.end, 0.2 )
model.add_transition( state2, model.end, 0.2 )


model.bake()

sequence = model.sample()

print(sequence)
print(model.log_probability(sequence))
print(math.exp(model.log_probability(sequence)/len(sequence)))

print (model.forward( sequence )[ len(sequence), model.end_index ])

model.fit([sequence])

print(model.log_probability(sequence))
print(math.exp(model.log_probability(sequence)/len(sequence)))


