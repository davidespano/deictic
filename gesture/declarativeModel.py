from enum import Enum
from dataset import *
from model import *
from topology import *

class ClassifierFactory:
    def __init__(self):
        self.line = None
        self.arc_cw = None
        self.arc_ccw = None
        self.scale = 100
        self.states = 6
        self.spu = 40  # samples per unit
        self.seq_edges = []

    def setClockwiseArcSamplesPath(self, path):
        self.arc_cw = path

    def setCounterClockwiseArcSamplesPath(self, path):
        self.arc_ccw = path

    def setLineSamplesPath(self, path):
        self.line = path

    def createClassifier(self, exp):
        # center and normalise the ideal model definition
        processor = ModelPreprocessor(exp)
        transform1 = CenteringTransform()
        transform2 = NormaliseLengthTransform(axisMode=True)
        processor.transforms.addTranform(transform1)
        processor.transforms.addTranform(transform2)
        processor.preprocess()
        #exp.plot()
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
            return OpEnum.Point, None

        if isinstance(exp, Line):
            alpha = math.acos(Geometry2D.getCosineFromSides(exp.dx, exp.dy)) # rotation with respect to left-to-right line
            if exp.dy < 0:
                alpha = -alpha
            distance = Geometry2D.distance(0,0, exp.dx, exp.dy)
            samples = round(distance * self.spu)
            n_states = round(distance * self.states)

            dataset = CsvDataset(self.line) # reads the raw data

            self.transformPrimitive(dataset, distance, alpha, startPoint, samples) # creates the primitive transforms

            hmm = self.createCleanLine(str(exp), alpha, distance * self.scale / samples, n_states)  # creates empty HMM
            samples = dataset.applyTransforms()
            if d:
                self.debugPlot(samples, exp)
            hmm.fit(samples, use_pseudocount=True) # trains it with the transformed samples

            startPoint[0] += exp.dx
            startPoint[1] += exp.dy

            return OpEnum.Line, [(hmm, None)]

        if isinstance(exp, Arc):
            if exp.cw:
                if exp.dx >= 0:
                    if exp.dy >= 0:
                        alpha = 0.5 * math.pi
                        center = [-0.5, 0]
                        gamma = 0
                    else:
                        alpha = 0;
                        center = [0,-0.5]
                        gamma = 1.5 * math.pi
                else:
                    if exp.dy >= 0:
                        alpha = math.pi;
                        center = [0, 0.5]
                        gamma = math.pi
                    else:
                        alpha = -0.5 * math.pi
                        center = [-0.5, 0]
                        gamma = 0.5 * math.pi

            else:
                if exp.dx >= 0:
                    if exp.dy >= 0:             #1
                        alpha = -0.5 * math.pi
                        center = [0, 0.5]
                        gamma = 1.5 * math.pi
                    else:                       #2
                        alpha = math.pi;
                        center = [0.5, 0]
                        gamma = math.pi
                else:
                    if exp.dy >= 0:             #3
                        alpha = 0;
                        center = [-0.5, 0]
                        gamma = 0
                    else:
                        alpha = 0.5 * math.pi
                        center = [0, -0.5]
                        gamma = 0.5 * math.pi

            distance = abs(0.5 * math.pi * exp.dx)
            samples = round(distance * self.spu)
            n_states = round(distance * self.states)

            if exp.cw:
                dataset = CsvDataset(self.arc_cw)
            else:
                dataset = CsvDataset(self.arc_ccw)

            self.trasformArcPrimitive(dataset, abs(exp.dx), alpha, center, startPoint, samples)

            hmm = self.createCleanArc(str(exp), alpha, abs(exp.dx) * self.scale, n_states, exp.cw)

            samples = dataset.applyTransforms()
            if d:
                self.debugPlot(samples, exp)
            hmm.fit(samples, use_pseudocount=True)  # trains it with the transformed samples

            startPoint[0] += exp.dx
            startPoint[1] += exp.dy

            return OpEnum.Arc, [(hmm, None)]


        # -----------------------------------------------
        #       composite terms
        # -----------------------------------------------
        if isinstance(exp, CompositeExp):
            oldPoint = [startPoint[0], startPoint[1]]
            expTypeLeft,  hmmLeft  = self.parseExpression(exp.left, startPoint)
            self.updateStartPoint(startPoint, oldPoint, exp.op)
            expTypeRight, hmmRight = self.parseExpression(exp.right, startPoint)

            expType = exp.op

            if (expTypeRight == expType and expType == expTypeLeft) \
                    or OpEnum.isGround(expTypeLeft) \
                    or OpEnum.isGround(expTypeRight):
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

        if isinstance(exp, IterativeExp):
            expType, hmm = self.parseExpression(exp.exp, startPoint)
            operand, seq_edge = self.createHMM(str(exp.exp), expType, hmm)
            return expType, [(hmm, seq_edge)]

        return None

    def trasformArcPrimitive(self, dataset, radius, alpha, center, startPoint, samples):
        # applying transforms for fitting
        centering = CenteringTransform()  # centering the line
        normalise = NormaliseLengthTransform(axisMode=True)  # normalise the length
        origin = TranslateTransform(t=[0.5, 0.5])
        rotate = RotateTransform(theta=alpha,
                                 unit=RotateTransform.radians)
        centerTranslate = TranslateTransform(t=[center[0] * self.scale, center[1] * self.scale])
        scale = ScaleDatasetTransform(scale=self.scale * radius)  # adjust the size
        resample = ResampleInSpaceTransform(samples=samples)  # resampling
        translate = TranslateTransform(
            t=[startPoint[0] * self.scale, startPoint[1] * self.scale])  # position the segment at its starting point

        dataset.addTransform(centering)
        dataset.addTransform(normalise)
        dataset.addTransform(origin)
        dataset.addTransform(scale)
        dataset.addTransform(rotate)
        dataset.addTransform(centerTranslate)
        dataset.addTransform(resample)
        dataset.addTransform(translate)
        return None


    def transformPrimitive(self, dataset, distance, alpha, startPoint, samples):
        # applying transforms for fitting the expression
        centering = CenteringTransform()  # centering the line
        normalise = NormaliseLengthTransform(axisMode=False)  # normalise the length
        origin = TranslateTransform(t=[0.5, 0])  # put the first sample in the origin, the line goes left-to-right
        rotate = RotateTransform(theta=alpha,
                                 unit=RotateTransform.radians)  # rotate the samples for getting the right slope
        scale = ScaleDatasetTransform(scale=distance * self.scale)  # adjust the size
        resample = ResampleInSpaceTransform(samples=samples)  # resampling
        translate = TranslateTransform(
            t=[startPoint[0] * self.scale, startPoint[1] * self.scale])  # position the segment at its starting point

        dataset.addTransform(centering)
        dataset.addTransform(normalise)
        dataset.addTransform(origin)
        dataset.addTransform(scale)
        dataset.addTransform(rotate)
        dataset.addTransform(resample)
        dataset.addTransform(translate)

    def updateStartPoint(self, startPoint, oldPoint, op):
        # for parallel and choice, the starting point has to go back to the old position
        if op == OpEnum.Parallel or op == OpEnum.Choice:
            startPoint[0] = oldPoint[0]
            startPoint[1] = oldPoint[1]


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

    def createCleanArc(self, name, alpha, radius, n_states, cw):
        topology_factory = HiddenMarkovModelTopology()  # Topology
        distributions = []

        step = 0.5 * math.pi / n_states

        beta = 0
        for i in range(0, n_states):
            a = cos(alpha + beta) * radius
            b = sin(alpha + beta) * radius

            gaussianX = NormalDistribution(a, self.scale * 0.01)
            gaussianY = NormalDistribution(b, self.scale * 0.01)
            #gaussianX = NormalDistribution(a,  0.01)
            #gaussianY = NormalDistribution(b,  0.01)
            distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY]))

            if cw:
                beta -= step
            else:
                beta += step

        return topology_factory.forward(name, n_states, distributions)



        return None

    def debugPlot(self, samples, exp):
        plt.axis("equal")
        for sample in samples:
            plt.plot(sample[:, 0], sample[:, 1], marker='.')
        plt.title(str(exp))
        plt.show()

    def createHMM(self, name, operator, operands):
        edges = []
        models = []
        for model, edge in operands:
            edges.append(edge)
            models.append(model)

        if operator == OpEnum.Sequence: # create a sequence
            sequence, seq_edges = HiddenMarkovModelTopology.sequence(operands = models, gt_edges= edges);
            sequence.name = name
            return sequence, seq_edges

        if operator == OpEnum.Parallel: # create a parallel gesture
            parallel = OpEnum[0]

            for i in range(1, len(models)):
                parallel, edges = HiddenMarkovModelTopology.parallel(parallel, models[i], edges)
            parallel.name = name
            return parallel, edges

        if operator == OpEnum.Choice: # create a choice
            choice, seq_edges = HiddenMarkovModelTopology.choice(operands=models, gt_edges=edges);
            choice.name = name
            return choice, seq_edges

        if operator == OpEnum.Disabling: # create a disabling
            disabling = models[0]

            for i in range(1, len(models)):
                disabling, edges = HiddenMarkovModelTopology.disabling(disabling, models[i], edges)
            disabling.name = name
            return disabling, edges

        if operator == OpEnum.Iterative:
            model = models[0]
            iterative, edges = HiddenMarkovModelTopology.iterative(model, edges)
            return iterative, edges

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
                points[i][2].x = transformed[i][0]
                points[i][2].y = transformed[i][1]
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