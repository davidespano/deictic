from enum import Enum
from dataset import *
from model import *
from topology import *
# config
from config import *
from gesture import ClassifierFactory, ModelPreprocessor

class ClassifierFactoryExt(ClassifierFactory):

    def createClassifier(self, exp):
        """

        :param exp:
        :return:
        """
        # center and normalise the ideal model definition
        processor = ModelPreprocessor(exp)
        transform1 = CenteringTransform()
        transform2 = NormaliseLengthTransform(axisMode=True)
        # transform3 = RotateCenterTransform(traslationMode=True)####
        processor.transforms.addTranform(transform1)
        # processor.transforms.addTranform(transform3)####
        processor.transforms.addTranform(transform2)
        processor.preprocess()
        # exp.plot()
        startPoint = [0, 0]
        self.strokeList = []
        self.stroke = -1
        self.parseStrokes(exp)
        self.stroke = -1
        # set distance
        self.distance = exp.get_distance()
        expType, operands = self.parseExpression(exp, startPoint)
        return self.createHMM(str(exp), expType, operands)

    def parseExpression(self, exp, startPoint, d=False):
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

        # Line 2D
        if isinstance(exp, Line):
            # num states
            prop_states = round((exp.get_distance() * self.states) / self.distance)
            n_states = prop_states if prop_states >=2 else 2
            # num samples
            samples = round(exp.get_distance() * self.spu)
            # alpha
            alpha = math.acos(
                Geometry2D.getCosineFromSides(exp.dx, exp.dy))  # rotation with respect to left-to-right line
            if exp.dy < 0:
                alpha = -alpha

            #### debug
            #print("Expr: "+str(exp))
            #print("Distance Total: "+str(self.distance))
            #print("Distance: "+str(exp.get_distance())+" - num_states: "+str(n_states) +" - samples: "+ str(samples))
            ####

            # get samples
            dataset = CsvDataset(self.line)# reads the raw data
            self.transformLinePrimitive(dataset, exp.get_distance(), alpha, startPoint, samples) # creates the primitive transforms
            if self.type == TypeRecognizer.online:
                self.transformOnline(dataset)
            samples = [sequence[0] for sequence in dataset.applyTransforms()]

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

            # num states
            prop_states = round((exp.get_distance() * self.states) / self.distance)
            n_states = prop_states if prop_states >= 2 else 2
            # num samples
            samples = round(exp.get_distance() * self.spu)

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
