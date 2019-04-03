from enum import Enum
from dataset import *
from model import *
from topology import *
# config
from config import *
# parsing
#from online.parsing_trajectory.trajectory_parsing import Parsing

class ClassifierFactory:
    """

    """

    # constructor
    def __init__(self, type=TypeRecognizer.offline, num_states=6, spu=20):
        # check parameters
        if not isinstance(type, TypeRecognizer):
            raise TypeError
        if not isinstance(num_states, int):
            raise TypeError
        if not isinstance(spu, (int, float)):
            raise TypeError
        # initialization parameters
        self.line = Config.trainingDir  # path training files - line
        self.arc_cw = Config.arcClockWiseDir  # path training files - arc clock wise
        self.arc_ccw = Config.arcCounterClockWiseDir  # path training files - arc counter clock wise
        self.type = type  # (offline or online?)
        self.states = num_states  # state numbers of model
        self.spu = spu  # samples per unit
        self.scale = 100
        self.seq_edges = []
        self.stroke = -1
        self.strokeList = []
        self.__chars = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'O']

    # public methods
    def setLineSamplesPath(self, path):
        self.line = path
    def setClockwiseArcSamplesPath(self, path):
        self.arc_cw = path
    def setCounterClockwiseArcSamplesPath(self, path):
        self.arc_ccw = path


    def createClassifier(self, exp):
        """

        :param exp:
        :return:
        """
        # center and normalise the ideal model definition
        processor = ModelPreprocessor(exp)
        transform1 = CenteringTransform()
        transform2 = NormaliseLengthTransform(axisMode=True)
        #transform3 = RotateCenterTransform(traslationMode=True)####
        processor.transforms.addTranform(transform1)
        #processor.transforms.addTranform(transform3)####
        processor.transforms.addTranform(transform2)
        processor.preprocess()
        #exp.plot()
        startPoint = [0,0]
        self.strokeList = []
        self.stroke = -1
        self.parseStrokes(exp)
        self.stroke = -1
        expType, operands = self.parseExpression(exp, startPoint)
        return self.createHMM(str(exp), expType, operands)

    def parseStrokes(self, exp):
        """

        :param exp:
        :return:
        """
        if isinstance(exp, Point):
            self.stroke += 1
            self.strokeList.append(str(self.stroke + 1))

        if isinstance(exp, CompositeExp):
            if not exp.left is None:
                self.parseStrokes(exp.left)
            if not exp.right is None:
                self.parseStrokes(exp.right)

    def parseExpression(self, exp, startPoint, d = False):
        """

        :param exp:
        :param startPoint:
        :param d:
        :return:
        """

        # -----------------------------------------------
        #       ground terms
        # -----------------------------------------------
        # Point 2D
        if isinstance(exp, Point):
            self.stroke += 1
            startPoint[0] = exp.x
            startPoint[1] = exp.y
            return OpEnum.Point, None

        # Point 3D
        if isinstance(exp, Point3D):
            self.stroke += 1
            startPoint[0] = exp.x
            startPoint[1] = exp.y
            startPoint[2] = exp.z
            return OpEnum.Point3D, None

        # Line 2D
        if isinstance(exp, Line):
            alpha = math.acos(Geometry2D.getCosineFromSides(exp.dx, exp.dy)) # rotation with respect to left-to-right line

            if exp.dy < 0:
                alpha = -alpha
            distance = Geometry2D.distance(0,0, exp.dx, exp.dy)
            samples = round(distance * self.spu)
            n_states = round(distance * self.states + 0.5)

            # get samples
            dataset = CsvDataset(self.line) # reads the raw data
            self.transformLinePrimitive(dataset, distance, alpha, startPoint, samples) # creates the primitive transforms
            if self.type == TypeRecognizer.online:
                self.transformOnline(dataset)
            samples = [sequence[0] for sequence in dataset.applyTransforms()]

            # add more samples
            #for i in range(3):
            #    samples.extend(samples)

            ### Debug ###
            if d:
                # # dataset example
                base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/raw/"
                dataset_original = CsvDataset(base_dir + "caret/")
                kalmanTransform = KalmanFilterTransform()
                parse = ParseSamples()
                dataset_original.addTransform(kalmanTransform)
                dataset_original.addTransform(parse)
                sequences = dataset_original.applyTransforms()
                sequence = sequences[0][0]
                s = sequence.getPointsSequence()
                l = sequence.getLabelsSequence()
                # plot original sequence
                original = plt.plot(s[:, 0], s[:, 1]+150, color='r')
                for i in range(len(l) - 1):
                   plt.annotate(l[i], (s[i, 0], s[i, 1]+150))
                ########### Plotting ########################
                sample = samples[5]
                sequence = sample.getPointsSequence()
                label_list = sample.getLabelsSequence()
                # plot original sequence
                print(len(sequence))
                syntethic = plt.plot(sequence[:, 0], sequence[:, 1], color='b')
                # label
                for i in range(len(label_list) - 1):
                    plt.annotate(label_list[i], (sequence[i, 0], sequence[i, 1]))
                plt.axis('equal')
                plt.show
            ### Debug ###

            # create hmm
            hmm = self.createCleanLine(str(exp), startPoint, exp.dx, exp.dy, self.scale, n_states)  # creates empty HMM
            # training
            hmm.fit(samples)#, use_pseudocount=True) # trains it with the transformed samples


            self.addStrokeIdDistribution(hmm)


            startPoint[0] += exp.dx
            startPoint[1] += exp.dy

            return OpEnum.Line, [(hmm, None)]
        # Arc
        if isinstance(exp, Arc):
            exp.dz = 0;
            if exp.cw:
                if exp.dx >= 0:
                    if exp.dy >= 0:
                        alpha = 0.5 * math.pi
                        center = [1, 0]
                        gamma = 0
                    else:
                        alpha = 0;
                        center = [0,-1]
                        gamma = 1.5 * math.pi
                else:
                    if exp.dy >= 0:
                        alpha = math.pi;
                        center = [0, 1]
                        gamma = math.pi
                    else:
                        alpha = -0.5 * math.pi
                        center = [-1, 0]
                        gamma = 0.5 * math.pi

            else:
                if exp.dx >= 0:
                    if exp.dy >= 0:             #1
                        alpha = -0.5 * math.pi
                        center = [0, 1]
                        gamma = 1.5 * math.pi
                    else:                       #2
                        alpha = math.pi;
                        center = [1, 0]
                        gamma = math.pi
                else:
                    if exp.dy >= 0:             #3
                        alpha = 0;
                        center = [-1, 0]
                        gamma = 0
                    else:
                        alpha = 0.5 * math.pi
                        center = [0, -1]
                        gamma = 0.5 * math.pi

            distance = abs(0.5 * math.pi * exp.dx)
            samples = round(distance * self.spu)
            n_states = round(distance * self.states + 0.5)

            if exp.cw:
                dataset = CsvDataset(self.arc_cw)
            else:
                dataset = CsvDataset(self.arc_ccw)

            # get samples
            translate = [startPoint[0] + center[0], startPoint[1] + center[1]]
            self.trasformArcPrimitive(dataset, startPoint, exp.dx, exp.dy, alpha, center, samples)
            if self.type == TypeRecognizer.online:
                self.transformOnline(dataset)
            samples = [sequence[0] for sequence in dataset.applyTransforms()]
            # add more samples
            # for i in range(3):
            #     samples.extend(samples)

            #if d:
            #    self.debugPlot(samples, exp)
            # create hmm
            hmm = self.createCleanArc(str(exp), startPoint, exp, self.scale, n_states)
            # training
            hmm.fit(samples)#, use_pseudocount=True)  # trains it with the transformed samples

            self.addStrokeIdDistribution(hmm)

            startPoint[0] += exp.dx
            startPoint[1] += exp.dy
            #startPoint[2] += exp.dz

            return OpEnum.Arc, [(hmm, None)]

        # todo - incomplete
        # Line 3D
        if isinstance(exp, Line3D):
            # Gets angles
            angles = Geometry3D.getCosineFromSides(exp.dx, exp.dy, exp.dz)

            #
            #if exp.dy < 0:
            #    alpha = -alpha

            # Computes the distance between origin and the passed vector. The distance value is used
            # to compute the correct number of samples and states.
            distance = Geometry3D.distance(0,0,0, exp.dx, exp.dy, exp.dz)
            samples = round(distance * self.spu)
            n_states = round(distance * self.states + 0.5)

            # Reads the raw data
            dataset = CsvDataset(self.line)

            # Creates the primitives transforms
            self.transformLine3DPrimitive(dataset, distance, angles, startPoint, samples)

            # Creates empty HMM
            hmm = self.createCleanLine3D(str(exp), startPoint, exp.dx, exp.dy, exp.dz, self.scale, n_states)
            samples = dataset.applyTransforms()

            if d:
                self.debugPlot(samples, exp)

            # Trains it with the transformed samples
            hmm.fit(samples, use_pseudocount=True)

            #
            self.addStrokeIdDistribution(hmm)

            #
            startPoint[0] += exp.dx
            startPoint[1] += exp.dy

            return OpEnum.Line, [(hmm, None)]

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
            elif expTypeLeft == expType:
                rightOperand, seq_edge_right = self.createHMM(str(exp.right), expTypeRight, hmmRight)
                hmmLeft.append((rightOperand, seq_edge_right))
                return expType, hmmLeft
            else:
                leftOperand, seq_edge_left = self.createHMM(str(exp.left), expTypeLeft, hmmLeft)
                rightOperand, seq_edge_right = self.createHMM(str(exp.right), expTypeRight, hmmRight)
                return expType, [(leftOperand, seq_edge_left), (rightOperand, seq_edge_right)]

        if isinstance(exp, IterativeExp):
            expType, hmm = self.parseExpression(exp.exp, startPoint)
            operand, seq_edge = self.createHMM(str(exp.exp), expType, hmm)
            return OpEnum.Iterative, [(operand, seq_edge)]

        return None
    def addStrokeIdDistribution(self, hmm):
        if len(self.strokeList) > 1:
            step = 0.5 / len(hmm.states) +1
            for i in range(0, len(hmm.states)):
                state = hmm.states[i]
                if not state.distribution is None:
                    x = state.distribution.distributions[0]
                    y = state.distribution.distributions[1]
                    #s = NormalDistribution(self.stroke + 1.0, (i +1 ) * step)
                    s = NormalDistribution(self.stroke + 1.0, 0.01)
                    state.distribution = IndependentComponentsDistribution([x, y, s], weights = [1,1, 10000])



    def transformLinePrimitive(self, dataset, distance, alpha, startPoint, samples):
        # applying transforms for fitting the expression
        # kalman
        kalman = KalmanFilterTransform()
        centering = CenteringTransform()  # centering the line
        normalise = NormaliseLengthTransform(axisMode=False)  # normalise the length
        origin = TranslateTransform(t=[0.5, 0])  # put the first sample in the origin, the line goes left-to-right
        rotate = RotateTransform(theta=alpha, unit=RotateTransform.radians)  # rotate the samples for getting the right slope
        scale = ScaleDatasetTransform(scale=distance * self.scale)  # adjust the size
        resample = ResampleInSpaceTransform(samples=samples)  # resampling
        #resample = ResampleTransform(delta=5)
        translate = TranslateTransform(t=[startPoint[0] * self.scale, startPoint[1] * self.scale])  # position the segment at its starting point
        # adding transforms to dataset
        #dataset.addTransform(kalman)
        dataset.addTransform(centering)
        dataset.addTransform(normalise)
        dataset.addTransform(origin)
        dataset.addTransform(scale)
        dataset.addTransform(rotate)
        dataset.addTransform(resample)
        dataset.addTransform(translate)

    def trasformArcPrimitive(self, dataset, startPoint, radiusX, radiusY, alpha, center, samples):
        # applying transforms for fitting
        # kalman
        kalman = KalmanFilterTransform()
        centering = CenteringTransform()  # centering the line
        normalise = NormaliseLengthTransform(axisMode=True)  # normalise the length
        origin = TranslateTransform(t=[0.5, 0.5])
        rotate = RotateTransform(theta=alpha, unit=RotateTransform.radians)
        centerTranslate = TranslateTransform(t=[center[0], center[1]])
        radiusScale = ScaleDatasetTransform(scale=[abs(radiusX), abs(radiusY)])
        startPointTranslate = TranslateTransform(t=startPoint)
        scale = ScaleDatasetTransform(scale=[self.scale, self.scale])  # adjust the size
        resample = ResampleInSpaceTransform(samples=samples)  # resampling
        #resample = ResampleTransform(delta=5)
        # adding transforms to dataset
        #dataset.addTransform(kalman)
        dataset.addTransform(centering)
        dataset.addTransform(normalise)
        dataset.addTransform(origin)
        dataset.addTransform(rotate)
        dataset.addTransform(centerTranslate)
        dataset.addTransform(radiusScale)
        dataset.addTransform(startPointTranslate)
        dataset.addTransform(scale)
        dataset.addTransform(resample)
    def transformOnline(self, dataset):
        # applying transforms for fitting
        parse = ParseSamples()
        dataset.addTransform(parse)




    # todo - incomplete
    def transformLine3DPrimitive(self, dataset, distance, angles, startPoint, samples):
        """
            Applying transforms for fitting the expression.
        :param dataset:
        :param distance:
        :param angles:
        :param startPoint:
        :param samples:
        :return:
        """
        # Centering the line
        centering = CenteringTransform()
        # Normalise the length
        normalise = NormaliseLengthTransform(axisMode=False)
        # Put the first sample in the origin, the line goes left-to-right
        origin = TranslateTransform(t=[0.5, 0])
        # Rotation, first through alpha_xy then alpha_xz
        rotate_z = RotateTransform(theta=angles[0],
                                 unit=RotateTransform.radians)  # rotate the samples for getting the right slope
        rotate_y = RotateTransform(theta=angles[1],
                                 unit=RotateTransform.radians)
        rotate_x = RotateTransform(theta=angles[2],
                                   unit=RotateTransform.radians)
        # Adjust the size
        scale = ScaleDatasetTransform(scale=distance * self.scale)
        # Resampling
        resample = ResampleInSpaceTransform(samples=samples)
        # Position the segment at its starting point
        translate = TranslateTransform(t=[startPoint[0] * self.scale, startPoint[1] * self.scale])

        # Apply transforms
        dataset.addTransform(centering)
        dataset.addTransform(normalise)
        dataset.addTransform(origin)
        dataset.addTransform(scale)
        dataset.addTransform(rotate_z)
        dataset.addTransform(rotate_y)
        dataset.addTransform(rotate_x)
        dataset.addTransform(resample)
        dataset.addTransform(translate)



    def updateStartPoint(self, startPoint, oldPoint, op):
        """

        :param startPoint:
        :param oldPoint:
        :param op:
        :return:
        """
        # for parallel and choice, the starting point has to go back to the old position
        if op == OpEnum.Parallel or op == OpEnum.Choice:
            startPoint[0] = oldPoint[0]
            startPoint[1] = oldPoint[1]

    def addStrokeId(self, samples):
        """

        :param samples:
        :return:
        """
        # TODO this part wastes a lot of memory
        new_samples = []
        if len(self.strokeList) > 1:
            for i in range(0, len(samples)):
                sample = samples[i]
                new_sample = []
                for j in range (0, len(sample)):
                    new_sample.append(numpy.append(sample[j], self.stroke + 1))
                new_samples.append(new_sample)
            return new_samples
        return samples





    def createStrokeDiscreteDistribution(self):
        """

        :return:
        """
        d = {}
        for i in range(0, len(self.strokeList)):
            p = 0
            if i == self.stroke:
                p = 1
            d[self.strokeList[i]] = p
        return DiscreteDistribution(d)

    def createCleanLine(self, name, startPoint, dx, dy, scale, num_states):
        """

        :param name:
        :param startPoint:
        :param dx:
        :param dy:
        :param scale:
        :param samples:
        :return:
        """
        topology_factory = HiddenMarkovModelTopology()  # Topology
        distributions = []

        # states
        if self.type == TypeRecognizer.offline:
            # offline
            step_x = dx / max(num_states - 1, 1)
            step_y = dy / max(num_states - 1, 1)
            for i in range(0, num_states):
                a = (startPoint[0] + (i * step_x)) * scale
                b = (startPoint[1] + (i * step_y)) * scale
                gaussianX = NormalDistribution(a, self.scale * 0.01)
                gaussianY = NormalDistribution(b, self.scale * 0.01)
                distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY]))
        else:
            # online
            distributions = self.__DiscreteDistribution(num_states, distributions)

        return topology_factory.forward(name, num_states, distributions)

    def createCleanArc(self, name, startPoint, exp, scale, num_states):
        """

        :param name:
        :param startPoint:
        :param exp:
        :param scale:
        :param n_states:
        :return:
        """
        topology_factory = HiddenMarkovModelTopology()  # Topology
        distributions = []

        if self.type == TypeRecognizer.offline:
            # offline
            step = 0.5 * math.pi / max(num_states - 1, 1)

            beta = 0
            alpha = 0

            # TODO this may be better coded
            if exp.cw:
                if exp.dy > 0:
                    if exp.dx > 0:
                        alpha = 0
                    else:
                        alpha = 0.5 * math.pi
                else:
                    if exp.dx > 0:
                        alpha = 1.5 * math.pi
                    else:
                        alpha = math.pi
            else:
                if exp.dy > 0:
                    if exp.dx > 0:
                        alpha = 0.5 * math.pi
                    else:
                        alpha = math.pi
                else:
                    if exp.dx > 0:
                        alpha = 0
                    else:
                        alpha = 1.5 * math.pi
            beta = alpha + math.pi
            for i in range(0, num_states):
                a = (cos(beta) + cos(alpha)) * abs(exp.dx) + startPoint[0]
                b = (sin(beta) + sin(alpha)) * abs(exp.dy) + startPoint[1]
                gaussianX = NormalDistribution(a * scale, self.scale * 0.01)
                gaussianY = NormalDistribution(b * scale, self.scale * 0.01)
                if exp.cw:
                    beta -= step
                else:
                    beta += step
                distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY]))
        else:
            # online
            distributions = self.__DiscreteDistribution(num_states, distributions)

        return topology_factory.forward(name, num_states, distributions)

    # todo - incomplete
    def createCleanLine3D(self, name, startPoint, dx, dy, dz, scale, samples):
        topology_factory = HiddenMarkovModelTopology()  # Topology
        distributions = []
        #
        step_x = dx / max(samples - 1, 1)
        step_y = dy / max(samples - 1, 1)
        step_z = dz / max(samples - 1, 1)
        #
        for i in range(0, samples):
            a = (startPoint[0] + (i * step_x)) * scale
            b = (startPoint[1] + (i * step_y)) * scale
            c = (startPoint[2] + (i * step_z)) * scale
            gaussianX = NormalDistribution(a, self.scale * 0.01)
            gaussianY = NormalDistribution(b, self.scale * 0.01)
            gaussianZ = NormalDistribution(c, self.scale * 0.01)
            distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY, gaussianZ]))
        return topology_factory.forward(name, samples, distributions)

    def __DiscreteDistribution(self, num_states=0, emissions=[]):
        for i in range(0, num_states):
            random.seed(datetime.datetime.now())
            distribution_values = numpy.random.dirichlet(numpy.ones(len(self.__chars)), size=1)[0]
            values = {self.__chars[index]: distribution_values[index] for index in range(0, len(self.__chars))}
            emissions.append(DiscreteDistribution(values))
        return emissions





    def debugPlot(self, samples, exp):
        """

        :param samples:
        :param exp:
        :return:
        """
        plt.axis("equal")
        for sample in samples:
            plt.plot(sample[:, 0], sample[:, 1], marker='.')
        plt.title(str(exp))
        plt.show()

    def createHMM(self, name, operator, operands):
        """

        :param name:
        :param operator:
        :param operands:
        :return:
        """


        edges = []
        models = []
        for model, edge in operands:
            edges.append(edge)
            models.append(model)

        if operator == OpEnum.Iterative:
            model = models[0]
            iterative, edges = HiddenMarkovModelTopology.iterative(model, edges)
            return iterative, edges

        # if the operand is only one, the expression corresponds to the operand HMM
        # e.g. a sequence with only one operand is the operand itself

        if len(operands) == 1:
            return operands[0]

        if operator == OpEnum.Sequence: # create a sequence
            sequence, seq_edges = HiddenMarkovModelTopology.sequence(operands = models, gt_edges= edges);
            sequence.name = name
            return sequence, seq_edges

        if operator == OpEnum.Parallel: # create a parallel gesture
            parallel = models[0]

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

        return None





    def plot(self):
        plt.show()







class ModelPreprocessor:
    """

    """

    def __init__(self, exp):
        """

        :param exp:
        """
        self.exp = exp
        self.transforms = CompositeTransform()

    def preprocess(self):
        """

        :return:
        """
        points = self.exp.to_point_sequence()
        transformed = self.transforms.transform(points)

        x = 0
        y = 0
        z = 0
        # update the expression terms
        for i in range(0,len(points)):
            #### 2D ####
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

            #### 3D ####
            elif isinstance(points[i][2], Point3D):
                points[i][2].x = transformed[i][0]
                points[i][2].y = transformed[i][1]
                points[i][2].z = transformed[i][2]
                x = transformed[i][0]
                y = transformed[i][1]
                z = transformed[i][2]

            elif isinstance(points[i][2], Line3D):
                points[i][2].dx = transformed[i][0] - x
                points[i][2].dy = transformed[i][1] - y
                points[i][2].dz = transformed[i][2] - z
                x = transformed[i][0]
                y = transformed[i][1]
                z = transformed[i][2]




    # class __DistributionPrimitive():
    #     # chars
    #     __chars = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'O', '0']
    #     # singleton
    #     __singleton = None
    #
    #     # public methods
    #     def getInstance(self):
    #         if self.singleton == None:
    #             self.__function= {
    #                 TypeRecognizer.online: self.__DiscreteDistribution,
    #                 TypeRecognizer.offline: self.__NormalDistribution,
    #                 Arc: self.__NormalDistributionArc,
    #                 Line: self.__NormalDistributionLine,
    #             }
    #         return self.singleton
    #
    #     def getDistribution(self, type, *args):
    #         return self.__function[type](*args)
    #
    #     # private methods
    #     def __DistributionDiscrete(self, num_states=0, *args):
    #         for i in range(0, num_states):
    #             random.seed(datetime.datetime.now())
    #             distribution_values = numpy.random.dirichlet(numpy.ones(len(self. __chars)), size=1)[0]
    #             values = {self.__chars[index]: distribution_values[index] for index in range(0, len(self.__chars))}
    #             distributions.append(DiscreteDistribution(values))
    #         return distributions
    #     def __NormalDistrbution(self, operator, num_states, *args):
    #         distributions = []
    #         for index in range(0, num_states):
    #             distributions.append(IndependentComponentsDistribution([self.__function[operator](index, *args)]))
    #         return distributions
    #     def __NormalDistributionArc(self, beta, alpha, exp, step, scale, startPoint):
    #         a = (cos(beta) + cos(alpha)) * abs(exp.dx) + startPoint[0]
    #         b = (sin(beta) + sin(alpha)) * abs(exp.dy) + startPoint[1]
    #         gaussianX = NormalDistribution(a * scale, scale * 0.01)
    #         gaussianY = NormalDistribution(b * scale, scale * 0.01)
    #         if exp.cw:
    #             beta -= step
    #         else:
    #             beta += step
    #         return gaussianX, gaussianY
    #     def __NormalDistributionLine(self, index, step_x, step_y, scale, startPoint):
    #         a = (startPoint[0] + (index * step_x)) * scale
    #         b = (startPoint[1] + (index * step_y)) * scale
    #         return NormalDistribution(a, scale * 0.01), NormalDistribution(b, scale * 0.01)