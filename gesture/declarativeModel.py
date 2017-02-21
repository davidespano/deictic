from enum import Enum
from dataset import CompositeTransform
from model import *
from topology import *

class ClassifierFactory:
    def __init__(self):
        self.line = None
        self.arc = None
        self.scale = 100
        self.spu = 8  # samples per unit
        self.seq_edges = []

    def setArcSamplesPath(self, path):
        self.arc = path

    def setLineSamplesPath(self, path):
        self.line = path




    def createClassifier(self, exp):
        processor = ModelPreprocessor(exp)
        transform1 = CenteringTransform()
        transform2 = NormaliseLengthTransform(axisMode=True)
        processor.transforms.addTranform(transform1)
        processor.transforms.addTranform(transform2)
        processor.preprocess()
        startPoint = [0,0]
        expType, operands = self.parseExpression(exp, startPoint)
        return self.createHMM(str(exp), expType, operands)

    def parseExpression(self, exp, startPoint, d = False):

        # -----------------------------------------------
        #       ground terms
        # -----------------------------------------------
        if isinstance(exp, Point):
            startPoint[0] = exp.x
            startPoint[1] = exp.y
            return ExpEnum.Point, None

        if isinstance(exp, Line):
            alpha = math.acos(Geometry2D.getCosineFromSides(exp.dx, exp.dy)) # rotation with respect to left-to-right line
            if exp.dy < 0:
                alpha = -alpha
            distance = Geometry2D.distance(0,0, exp.dx, exp.dy)
            samples = round(distance * self.spu)

            dataset = CsvDataset(self.line) # reads the raw data

            # applying transforms for fitting the expression
            transform1 = CenteringTransform() # centering the line
            transform2 = NormaliseLengthTransform(axisMode=False) # normalise the length
            transform3 = TranslateTransform(t=[0.5,0]) # put the first sample in the origin, the line goes left-to-right
            transform4 = RotateTransform(theta=alpha, unit=RotateTransform.radians) # rotate the samples for getting the right slope
            transform6 = ScaleDatasetTransform(scale=distance * self.scale) # adjust the size
            transform7 = ResampleInSpaceTransform(samples=samples) # resampling
            transform8 = TranslateTransform(t=[startPoint[0], startPoint[1]]) # position the segment at its starting point

            dataset.addTransform(transform2)
            dataset.addTransform(transform1)
            dataset.addTransform(transform3)
            dataset.addTransform(transform4)
            dataset.addTransform(transform6)
            dataset.addTransform(transform7)
            dataset.addTransform(transform8)

            hmm = self.createCleanLine(str(exp), alpha, distance * self.scale / samples, samples)  # creates empty HMM
            samples = dataset.applyTransforms()
            if d:
                plt.axis("equal")
                for sample in samples:
                    plt.plot(sample[:, 0], sample[:, 1],  marker='.')
                plt.title(str(exp))
                plt.show()
            hmm.fit(dataset.applyTransforms(), use_pseudocount=True) # trains it with the transformed samples

            startPoint[0] += exp.dx * self.scale
            startPoint[1] += exp.dy * self.scale

            return ExpEnum.Line, [(hmm, None)]

        # -----------------------------------------------
        #       composite terms
        # -----------------------------------------------
        if isinstance(exp, CompositeExp):
            # TODO handle the starting point according to the operator semantics
            expTypeLeft,  hmmLeft  = self.parseExpression(exp.left, startPoint)
            expTypeRight, hmmRight = self.parseExpression(exp.right, startPoint)

            expType = ExpEnum.fromOpEnum(exp.op)

            if (expTypeRight == expType and expType == expTypeLeft) \
                    or ExpEnum.isGround(expTypeLeft) \
                    or ExpEnum.isGround(expTypeRight):
                # we can combine the lists without creating the hmm yet
                if hmmLeft is None: # point on the left operand
                    return expType, hmmRight
                if hmmRight is None: # point on the right operand
                    return expType, hmmLeft
                for el in hmmRight:
                    hmmLeft.append(el)
                return expType, hmmLeft
            else:
                leftOperand, seq_edge_left = self.createHMM(str(exp.left), expTypeLeft, hmmLeft)
                rightOperand, seq_edge_right = self.createHMM(str(exp.right), expTypeRight, hmmRight)
                return expType, [(leftOperand, seq_edge_left), (rightOperand, seq_edge_right)]

        return  None


    def createCleanLine(self, name, alpha, step, n_states):
        topology_factory = HiddenMarkovModelTopology()  # Topology
        distributions = []

        step_x = step * cos(alpha)
        step_y = step * sin(alpha)
        for i in range(0, n_states):
            a = i * step_x
            b = i * step_y

            gaussianX = NormalDistribution(a, self.scale * 0.01)
            gaussianY = NormalDistribution(b, self.scale * 0.01)
            distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY]))

        return topology_factory.forward(name, n_states, distributions)

    def createHMM(self, name, operator, operands):
        edges = []
        models = []
        for model, edge in operands:
            edges.append(edge)
            models.append(model)

        if operator == ExpEnum.Sequence: # create a sequence
            sequence, seq_edges = HiddenMarkovModelTopology.sequence(operands = models, gt_edges= edges);
            sequence.name = name
            return sequence, seq_edges

        if operator == ExpEnum.Parallel: # create a parallel gesture
            parallel = models[0]

            for i in range(1, len(models)):
                parallel, edges = HiddenMarkovModelTopology.parallel(parallel, models[i], edges)
            parallel.name = name
            return parallel, edges

        if operator == ExpEnum.Choice: # create a choice
            choice, seq_edges = HiddenMarkovModelTopology.choice(operands=models, gt_edges=edges);
            choice.name = name
            return choice, seq_edges

        if operator == ExpEnum.Disabling: # create a disabling
            disabling = models[0]

            for i in range(1, len(models)):
                disabling, edges = HiddenMarkovModelTopology.disabling(disabling, models[i], edges)
            disabling.name = name
            return disabling, edges

        # TODO add iterative definition

        return None


class ModelPreprocessor:

    def __init__(self, exp):
        self.exp = exp
        self.transforms = CompositeTransform()

    def preprocess(self):
        points = self.exp.to_point_sequence()
        transformed = self.transforms.transform(points)

        x = 0
        y = 0
        # update the expression terms
        for i in range(0,len(points)):
            if isinstance(points[i][2], Point):
                points[i][2].x = transformed[i][0] - x
                points[i][2].y = transformed[i][1] - y
                x = transformed[i][0]
                y = transformed[i][1]

            elif isinstance(points[i][2], Line):
                points[i][2].dx = transformed[i][0] - x
                points[i][2].dy = transformed[i][1] - y
                x = transformed[i][0]
                y = transformed[i][1]

            elif isinstance(points[i][2], Arc):
                points[i][2].dx = transformed[i][0] - x
                points[i][2].dy = transformed[i][1] - y
                x = transformed[i][0]
                y = transformed[i][1]


class ExpEnum(Enum):
    Undef = -1
    Point = 0
    Line = 1
    Arc = 2
    Sequence = 3
    Choice = 4
    Disabling = 5
    Iterative = 6
    Parallel = 7

    @staticmethod
    def fromOpEnum(opEnum):
        if opEnum == OpEnum.Sequence:
            return ExpEnum.Sequence
        if opEnum == OpEnum.Parallel:
            return ExpEnum.Parallel
        if opEnum == OpEnum.Choice:
            return ExpEnum.Choice
        return ExpEnum.Undef

    @staticmethod
    def isGround(opEnum):
        return opEnum == ExpEnum.Point or opEnum == ExpEnum.Line or opEnum == ExpEnum.Arc