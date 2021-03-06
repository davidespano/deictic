/**
 * @namespace Input
 * @description Contains the classes for managing the user's input. It covers both touch and mouse input with a
 * simplified and unified API. In addition, it contains the classes for recognizing gestures with the tree approaches
 * covered in the user test: heuristics, machine learning and deictic.
 */
(function (Input, undefined) {


    /**
     * @class Event
     * @memberOf Input
     * @description Creates a generic event
     * @classdesc Represent a generic user's input event. It manages subscription and unsubscription by event
     * handlers and the notification of the event.
     */
    var Event = function () {


        this.callback = [];

        /**
         * @public
         * @instance
         * @memberOf Input.Event
         * @method add
         * @description Adds an handler for the input event
         * @param {function} handler - the handler to be added
         */
        this.add = function (handler) {
            this.callback.push(handler);
        };

        /**
         * @public
         * @instance
         * @memberOf Input.Event
         * @method remove
         * @description Removes an handler for the input event
         * @param {function} handler - the handler to be removed
         */
        this.remove = function (handler) {
            var index = this.callback.indexOf(handler);
            if (index > -1) {
                this.callback.splice(index, 1);
            }
        };

        /**
         * @public
         * @instance
         * @memberOf Input.Event
         * @method trigger
         * @description Triggers the notification of the input event
         * @param {object} evt  - the event arguments
         */
        this.trigger = function (evt) {
            this.callback.forEach(function (l) {
                l(evt);
            });
        };
    };
    Input.Event = Event;

    /**
     * @class StrokeInput
     * @memberOf Input
     * @description Creates a stroke input manager
     * @classdesc Simplified manager for the user's stroke input, which may be received from both touch and mouse
     * events. It provides a unified API for writing device agnostic recognition algorithms.
     * @tutorial Registering strokes
     */
    var StrokeInput = function (layer) {
        /**
         * @public
         * @member onStrokeBegin
         * @memberOf Input.StrokeInput
         * @instance
         * @description Notifies the beginning of a new stroke. The event object contains a
         * {@link Input~StrokeChangeDescriptor} in the d property.
         * @type {Input.Event}
         * @see {Input~StrokeChangeDescriptor}
         * @example
         * // input is a Input.StrokeInput
         * // register an anonymous handler to the stroke begin event
         * input.onStrokeBegin.add(function(event){
         *      // sample handler body, it reads the coordinates of the last stroke point
         *      // the event object contains a StrokeChangeDescriptor in the d property
         *      var currentX = event.d.x;
         *      var currentY = event.d.y;
         * });
         */
        this.onStrokeBegin = new Event();
        /**
         * @public
         * @member onStrokeChange
         * @memberOf Input.StrokeInput
         * @instance
         * @description Notifies that the current stroke has changed, adding a new point to the stroke.
         * The event object contains a {@link Input~StrokeChangeDescriptor} in the d property.
         * @type {Input.Event}
         * @see {Input~StrokeChangeDescriptor}
         * @example
         * // input is a Input.StrokeInput
         * // register an anonymous handler to the stroke change event
         * input.onStrokeChange.add(function(event){
         *      // sample handler body, it computes the difference in the X and Y coordinates between the current
         *      // and the previous recorded stroke point, in grid cell coordinates.
         *      // Note that the event object contains a StrokeChangeDescriptor in the d property.
         *      var previous = event.d.buffer[event.d.buffer.length - 2];
         *      var diffX = event.d.x - previous.x;
         *      var diffY = event.d.y - previous.y;
         * });
         */
        this.onStrokeChange = new Event();
        /**
         * @public
         * @member onStrokeEnd
         * @memberOf Input.StrokeInput
         * @instance
         * @description Notifies that the current stroke is completed.
         * The event object contains a {@link Input~StrokeChangeDescriptor} in the d property.
         * @type {Input.Event}
         * @see {Input~StrokeChangeDescriptor}
         * @example
         * // input is a Input.StrokeInput
         * // register an anonymous handler to the stroke end event
         * input.onStrokeEnd.add(function(event){
         *      // sample handler body, it computes the difference in the X and Y coordinates between the current
         *      // and the previous recorded stroke point, in grid cell coordinates.
         *      // Note that the event object contains a StrokeChangeDescriptor in the d property.
         *      var previous = event.d.buffer[event.d.buffer.length - 2];
         *      var diffX = event.d.x - previous.x;
         *      var diffY = event.d.y - previous.y;
         * });
         */
        this.onStrokeEnd = new Event();

        this._layer = layer;
        this._buffer = [];
        this._recording = false;
        this._box = null;
        this._debounce = 3;
        this._count = 0;

        var _self = this;
        // attach the input source
        this._layer.on('mousedown touchstart', function (event) {
            _self._recording = true;
            _self._box = new Point2D(event.target.attrs.x, event.target.attrs.y);
            _self._buffer.splice(0, _self._buffer.length);
            var point = boxCoordinates(event);
            _self._buffer.push(boxCoordinates(event));
            event.d = createEvent(point);
            _self.onStrokeBegin.trigger(event);
        });


        this._layer.on('mousemove touchmove', function (event) {
            if (_self._recording === true) {
                if (_self._count === _self._debounce) {
                    _self._count = 0;
                    var point = boxCoordinates(event);
                    _self._buffer.push(boxCoordinates(event));
                    event.d = createEvent(point);
                    _self.onStrokeChange.trigger(event);
                } else {
                    _self._count++;
                    return;
                }
            }
        });

        this._layer.on('mouseup touchend', function (event) {
            _self._recording = false;
            var point = boxCoordinates(event);
            _self._buffer.push(boxCoordinates(event));
            event.d = createEvent(point);
            _self.onStrokeEnd.trigger(event);
            _self._box = null;

        });

        var boxCoordinates = function (kevent) {
            var point;
            switch (kevent.type) {
                case 'touchmove':
                case 'touchend':
                    point = new Point(
                        kevent.evt.changedTouches[0].clientX,
                        kevent.evt.changedTouches[0].clientY
                    );
                    break;
                default:
                    point = new Point2D(kevent.evt.x, kevent.evt.y);
                    break;

            }
            //console.log('punto: ' + point.x + ', ' + point.y +
            //    ' l: ' + _self._buffer.length +
            //    ' k: ' + kevent.target.attrs.x + ', ' + kevent.target.attrs.y);
            return new Point2D(point.x - _self._box.x, point.y - _self._box.y);
        };

        var createEvent = function (point) {
            /**
             * @description Represent the stroke change description
             * @typedef  {Object} Input~StrokeChangeDescriptor
             * @property {number} x - the X coordinate of the current point, expressed in cell coordinates (distance
             * from the top-left corner of the cell that contains the point)
             * @property {number} y - the Y coordinate of the current point, expressed in cell coordinates (the distance
             * from the top-left corner of the cell that contains the point)
             * @property {number} bX - the X coordinate of the cell containing the stroke, in the scene coordinates
             * (the distance from the top-left corner of the grid map)
             * @property {number} bY - the Y coordinate of the cell containing the stroke, in the scene coordinates
             * (the distance from the top-left corner of the grid map)
             * @property {Array<Input.Point2D>} buffer - a polyline representing the entire stroke, expressed as an
             * array of {@link Input.Point2D}.
             * @example
             * // d is a StrokeChangeDescriptor
             * // get the current stroke point in scene (grid map) coordinates
             * var sX = d.x + d.bX;
             * var sY = d.y + d.bY;
             *
             * // get the coordinates of the first stroke point in cell coordinates
             * var firstX = d.buffer[0].x;
             * var firstY = d.buffer[0].y;
             *
             * // get the coordinates of the previous stroke point in scene (grid map) coordinates
             * // note that d.buffer[d.buffer.length -1].x === d.x
             * var prevX = d.buffer[d.buffer.length - 2].x + d.bX;
             * var prevY = d.buffer[d.buffer.length - 2].y + d.bY;
             */
            return {
                x: point.x,
                y: point.y,
                bX: _self._box.x,
                bY: _self._box.y,
                buffer: _self._buffer
            };
        }


    };
    Input.StrokeInput = StrokeInput;


    /**
     * @class Point2D
     * @memberOf Input
     * @description Create a point with 2D coordinates
     * @classdesc Represents a point in a 2D coordinate system.
     */
    var Point2D = function (x, y) {
        /**
         * @instance
         * @member {number} x - the X coordinate of the point
         * @memberOf Input.Point2D
         */
        this.x = x;
        /**
         * @instance
         * @member {number} y - the Y coordinate of the point
         * @memberOf Input.Point2D
         */
        this.y = y;
        this.X = x;
        this.Y = y;
    };
    Input.Point2D = Point2D;

    /**
     * @class AngleFSM
     * @memberOf Input
     * @classdesc Creates a Finite State Machine FSM) that recognizes gestures according to the direction of
     * the stroke movements, applying an heuristic on the stroke change angle.
     * We define  the <em>movement vector</em> as the displacement
     * between the current and the previous stroke position. We translate it into the origin and we calculate the angle
     * it defines by the vector and the X axis for obtaining the <em>movement vector angle</em>.
     * <br/>
     * The FSM is defined through a list of angle ranges (minimum and maximum). Each range is associated to a
     * FSM state.
     * <br/>
     * The FSM stays in the current state as long it receives movement vector angles contained into
     * the specified range. It goes in the next state (if any) when it receives a certain number of movement vector
     * angles contained into the range associated to the next state.
     * It is possible to control such number through the tolerance member (default: 3).
     * <br/>
     * If it receives a number of movement vector angles higher than the tolerance threshold that were not contained
     * in the current state or the next state range, the FSM goes in an error state.
     * @description Creates a Finite State Machine for recognizing gestures with an heuristic approach.
     * @tutorial Recognizing gestures with Finite State Machines
     */
    var AngleFSM = function () {
        var _self = this;


        /**
         * @public
         * @instance
         * @method init
         * @description Inits the {@link Input.AngleFSM} instance with the specified configuration.
         * @memberOf Input.AngleFSM
         * @param {Array<Input~AngleFSMState>} stateDesc - a configuration containing the array of ranges defining
         * the FSM states
         * @param {number} [tolerance = 3] - the number of movement vector angles to be counted before firing the
         * transition to the next state or to the error state, according to the range check.
         * @example
         * var angleFSM = Input.AngleFSM();
         * // configures angleFSM for recognizing a V gesture.
         * // We specify the ideal slope plus or minus 20 degrees.
         * angleFSM.init([
         *    {min: 315 -20, max: 315 + 20},
         *    {min: 45 - 20, max: 45 + 20}
         *  ], 5);
         */
        this.init = function (stateDesc, tolerance) {

            /**
             * @instance
             * @member {number} state - the current FSM state. Values greater than or equal to
             * zero represent the state index in the {@link Input.AngleFSM#states|states} array. Values less than
             * zero represent the error state.
             * @memberOf Input.AngleFSM
             */
            _self.state = -1;
            _self.error = 0;
            _self.next = 0;
            /**
             * @description Represents an {@link Input.AngleFSM} state
             * @typedef  {Object} Input~AngleFSMState
             * @property {number} min - the minimum acceptable angle of the direction vector in degrees
             * @property {number} max - the maximum acceptable angle of the direction vector in degrees
             * @example
             * // recognizes the movement vector angles between 0 and 45 degrees
             * var range = {min: 0, max: 45};
             *
             * // recognizes the movement vector between -20 and 20 degrees
             * var range = {min: -20, max: 20};
             */
            /**
             * @instance
             * @member {Array<Input~AngleFSMState>} states - the range list defining the FSM states
             * @memberOf Input.AngleFSM
             */
            _self.states = stateDesc;
            /**
             * @instance
             * @member {number} tolerance - the number of movement vector angles to be counted before firing the
             * transition to the next state or to the error state, according to the range check.
             * @memberOf Input.AngleFSM
             */
            _self.tolerance = tolerance != null ? tolerance : 3;
        };

        /**
         * @public
         * @instance
         * @method restart
         * @description Resets the FSM to its initial state.
         * @memberOf Input.AngleFSM
         */
        this.restart = function () {
            _self.state = -1;
            _self.error = 0;
            _self.next = 0;
        };

        /**
         * @public
         * @instance
         * @method push
         * @description Feeds the FSM with a new movement vector. The next state will be updated according to the
         * received vector and the {@link Input.AngleFSM#tolerance|tolerance}.
         * @memberOf Input.AngleFSM
         * @param {Input.Point2D} current - the current stroke position
         * @param {Input.Point2D} previous - the previous stroke position
         * @example
         * // When receiveing an update from the StrokeInput input, we feed it
         * // to the AngleFSM fsm
         * input.onStrokeChange.add(function (event) {
         *     var current = event.d;
         *     var previous = event.d.buffer[event.d.buffer.length - 2];
         *     fsm.push(current, previous);
         * });
         *
         */
        this.push = function (current, previous) {
            var a = angle(current.x - previous.x, current.y - previous.y);
            if (isNaN(a)) {
                return;
            }
            switch (_self.state) {
                case -1: // start
                    _self.state++;
                    _self.next = 0;
                    _self.err = 0;
                    break;
                case -2: // not recognized
                    break;
                default:
                    if (_self.checkAngle(a, _self.states[_self.state])) {
                        _self.sampleOk();
                    } else {
                        if (_self.state + 1 < _self.states.length && _self.checkAngle(a, _self.states[_self.state + 1])
                        ) {
                            _self.sampleNext();
                        } else {
                            _self.sampleError();
                        }
                    }
                    break;
            }

            if (_self.error > _self.tolerance) {
                _self.state = -2;
            }

            if (_self.next > _self.tolerance) {
                _self.state++;
                _self.next = 0;
                _self.error = 0;
            }

            //console.log('angle: ' + a + ' state: ' + _self.state);
        };

        this.sampleError = function () {
            _self.error++;
            _self.next = 0;
        };

        this.sampleOk = function () {
            if (_self.error > 0) {
                _self.error--;
            }
            _self.next = 0;
        };

        this.sampleNext = function () {
            _self.next++;
        };

        this.checkAngle = function (angle, desc) {
            if (desc.min < 0 && desc.max < 0) {
                desc.min = desc.min + 360;
                desc.max = desc.max + 360;
            }

            if (desc.min > 0 && desc.max > 0) {
                return angle >= desc.min && angle <= desc.max;
            }

            return angle >= 0 && angle <= desc.max || angle >= desc.min && angle >= 360;
        };
    };
    Input.AngleFSM = AngleFSM;

    var angle = function (ax, ay) {
        var angle = Math.acos(ax / Math.sqrt(ax * ax + ay * ay));
        if (ay > 0) {
            angle = 2 * Math.PI - angle;
        }
        return angle * 180 / Math.PI;
    };
    Input.angle = angle;


    /**
     * @class Deictic
     * @memberOf Input
     * @classdesc A support for real-time gesture recognition with identification of the subparts. It allows
     * defining a gesture set composing points, lines and arc primitives. The recognition is robust with respect to
     * user's performance variability and, for each update received on the stroke path, it provides information on
     * the recognition probability of the whole gesture and the composed primitives. <br/>
     * A <a href="http://localhost:8000/basic/deictic_editor" target="_blank"> graphical modelling editor</a> for the gesture
     *  expressions is available. <br/>
     * The recognition is performed at server-side, but this hides completely the communication.
     * @description Creates an object for recognizing gestures with the Deictic approach.
     * @see Input~DeicticGesture
     * @tutorial Recognizing gestures with Deictic
     */
    var Deictic = function () {
        var _self = this;

        /**
         * @public
         * @instance
         * @method init
         * @description Inits the {@link Input.Deictic} instance with the specified configuration.
         * @memberOf Input.Deictic
         * @param {Array<Input~DeicticGesture>} gestures - the definition of the gesture set.
         * @returns true if Deictic has been correctly initialized, false otherwise.
         * @example
         * var deictic = new Input.Deictic();
         * // configures deictic for recognizing a V and a circle gesture (counter-clockwise)
         * deictic.init([
         *      {name: 'V', model: 'P(0,0) + L(2,-3) + L(2,3)'},
         *      {name: 'circle', model: 'P(0, 0) + A(-3, -3, false) + A(3, -3, false) + A(3, 3, false) + A(-3, 3, false)'}
         * ]);
         */
        this.init = function (gestures) {
            var result = true;
            console.log('Deictic: models init at server-side');
            /**
             * @description Represents a gesture in the  {@link Input.Deictic} format. <br/>
             * A <a href="http://localhost:8000/basic/deictic_editor" target="_blank"> graphical modelling editor</a> for the gesture
             *  expressions is available. <br/>
             * It is possible to combine the following 2D primitives:
             * <ul>
             *  <li> A <strong>2D point</strong>, denoted as <code>P(x,y)</code> where <code>x</code> and <code>y</code> are its
             *      coordinates. It represents the starting point of a stroke.
             *  </li>
             *  <li> A <strong>2D line</strong>, starting at the end of the previous primitive and moving <code>dx</code> along the X
             *      axis and <code>dy</code> along the Y axis, denoted as <code>L(dx, dy)</code>. It represents
             *      a movement along a linear path.
             *  </li>
             *  <li> A <strong>2D arc</strong>, starting at the end of the previous primitive, moving <code>dx</code> along the X
             *      axis and <code>dy</code> along the Y axis, clockwise if <code>cw === true</code> or
             *      counter-clockwise if <code>cw === false</code>. An arc is denoted as <code>A(dx, dy, cw)</code>.
             *      It represents a movement along a curved path, clockwise or counterclockwise.
             *  </li>
             *  </ul>
             *
             *  Applying the following composition operator it is possible to obtain complex gestures starting from
             *  primitives, declaring a modelling expression:
             *  <ul>
             *      <li>A <strong>sequence</strong>, represented by the symbol <code>+</code> indicating that
             *          the user has to complete first the left operand and then the right one. <code>A + B</code>
             *          denotes that, for completing the gesture, the user has first to perform <code>A</code>
             *          and then <code>B</code>
             *      </li>
             *      <li>
             *          A <strong>choice</strong>, represented by the symbol <code>|</code>, indicating that the
             *          the user has to complete either one operand or the other. <code>A | B</code>
             *          denotes that, for completing the gesture, the user may perform either <code>A</code>
             *          or <code>B</code>
             *      </li>
             *      <li>
             *          A <strong>parallel</strong> execution, represented by the symbol <code>*</code>,
             *          indicating that the user has to complete either both operands at (nearly) the same time.
             *          <code>A * B</code> denotes that, for completing the gesture, the user has to perform both
             *          <code>A</code> and <code>B</code>
             *      </li>
             *  </ul>
             *  A stroke always stars with a point. Then, it continues along a linear (line), curved (arc) or mixed
             *  (both of them) path. A multi-stroke gesture (e.g. performed with more than a finger) contains
             *  two or more point primitives. <br/>
             *  Please note that only the relative sizes among the primitives influence the recognition, since the
             *  dimensions are normalised by the underlying engine. Therefore the gesture
             *  <code>P(0,0) + L(0,1) + L(1,0)</code> is equivalent to <code>P(0,0) + L(0,5) + L(5,0)</code>, but it is
             *  different from <code>P(0,0) + L(0,1) + L(5,0)</code>, since the movement on the X  is 5 times
             *  the one on the Y.
             *
             * @typedef  {Object} Input~DeicticGesture
             * @property {string} name - the name associated to the gesture
             * @property {string} model - the gesture modelled with the Deictic primitives and operators.
             * @example
             * // single stroke horizontal line
             * var horizontal = {name: 'horizontal', model: 'P(0,0) + L(5, 0)'};
             *
             * // single stroke vertical line
             * var vertical = {name: 'vertical', model: 'P(0,0) + L(0, 5)'};
             *
             * // a V gesture
             * var v = {name: 'V', model: 'P(0,0) + L(2,-3) + L(2,3)'};
             *
             * // clockwise circle
             * var circleCw = {
             *     name: 'circleCw',
             *     model: 'P(0, 0) + A(3, -3, true) + A(-3, -3, true) + A(-3, 3, true) + A(3, 3, true)'
             * };
             *
             * // counterclockwise circle
             * var circleCcw = {
             *     name: 'circleCcw',
             *     model: 'P(0, 0) + A(-3, -3, false) + A(3, -3, false) + A(3, 3, false) + A(-3, 3, false)'
             * };
             *
             * // a question mark
             * var question = {
             *     name: 'question',
             *     model: 'P(0,0) + A(3,3, true) + A(3,-3, true) + A(-3,-3,true) + L(0,-4)'
             * };
             *
             * // a five-points star
             * var star = {name: 'star', model: 'P(0,0) + L(2,5) + L(2, -5) + L(-5, 3) + L(6,0) + L(-5, -3)'}
             */
            $.ajax({
                type: 'POST',
                url: "/basic/deictic_models",
                async: false,
                headers: {'X-CSRFToken': CSRF_TOKEN},
                data: JSON.stringify(gestures),
                dataType: 'json',
                contentType: "application/json; charset=utf-8",
                success: function (response) {
                    result = true;
                    console.log('Deictic: models init ok!');
                },
                error: function () {
                    result = false
                }

            });
            return result;
        };

        /**
         * @public
         * @instance
         * @method eval
         * @description Evaluates the current stroke and returns the probability associated to each gesture and
         * its sub-parts.
         * @memberOf Input.Deictic
         * @param {Array<Input.Point2D>} sequence - the current stroke represented as an array of {@link Input.Point2D}
         * @returns {Input~DeicticEvaluation} the evaluation result, null in case of errors.
         * @example
         * // Example 1: sample result dump
         * var result = {
         *      name: 'V', parts: [
         *          {name: 'P(0,0) + L(2,-3)', probability: 0.33},
         *          {name: 'P(0,0) + L(2,-3) + L(2,3)', probability: 0.75}
         *      ],
         *
         *      name: 'circleCw', parts: [
         *          {name: 'P(0, 0) + A(3, 3, true)', probability: 0},
         *          {name: 'P(0, 0) + A(3, 3, true) + A(3, -3, true)', probability: 0},
         *          {name: 'P(0, 0) + A(3, 3, true) + A(3, -3, true) + A(-3, -3, true)', probability: 0},
         *          {name: 'P(0, 0) + A(3, 3, true) + A(3, -3, true) + A(-3, -3, true) + A(-3, 3, true)', probability: 0},
         *      ]
         * };
         *
         * // Example 2: stroke update
         * // When receiving an update from the StrokeInput input,
         * // we evaluate it with Deictic. The object input is an
         * // Input.StrokeInput instance.
         * input.onStrokeChange.add(function (event) {
         *     var result = deictic.eval(event.d.buffer);
         *
         *     // check the probability of the second part of the V gesture
         *     var prob = 0;
         *     for(var i in result){
         *          var gesture = result[i];
         *          if(gesture.name === 'V'){
         *              prob = gesture.parts[1].probability;
         *          }
         *     }
         * });
         *
         */
        this.eval = function (sequence) {
            var result = false;
            $.ajax({
                type: 'POST',
                url: '/basic/deictic_eval',
                async: false,
                headers: {'X-CSRFToken': CSRF_TOKEN},
                data: JSON.stringify(sequence),
                dataType: 'json',
                contentType: "application/json; charset=utf-8",
                success: function (response) {
                    /**
                     * @description Represents the Deictic evaluation of a stroke.
                     * @typedef  {Object} Input~DeicticEvaluation
                     * @property {string} name - the name associated to the gesture
                     * @property {Array<Input~DeicticEvaluationPart>} parts - the Deictic evaluation associated to the
                     * different gesture parts.
                     *
                     */
                    /**
                     * @description Represents the Deictic probability evaluation of a gesture part
                     * @typedef {Object} Input~DeicticEvaluationPart
                     * @property {string} name - the name associated to the gesture part (e.g. the subpart expression)
                     * @property {number} probability - the probability associated to the gesture part
                     * @example
                     * // describes the probability of the first part of a V gesture
                     * var v1 = {name: 'P(0,0) + L(2,-3)', probability: 0.67};
                     */
                    result = response.result;
                },
                error: function () {
                    result = null
                }
            });
            return result;
        };

        /**
         * @public
         * @instance
         * @method recognisedGesture
         * @description A short-hand for getting the name of the gesture with the higher recognition probability.
         * @memberOf Input.Deictic
         * @param {Input~DeicticEvaluation} result - the Deictic evaluation of a stroke, as returned by the
         * {@link Input.Deictic#eval/eval} method.
         * @param {number} threshold - the minimum probability for considering the gesture as recognized.
         * @returns {string} the name of the gesture with the higher evaluation probability, null if all
         * gestures are below the threshold.
         * @example
         * // get the most likely gesture at the end of the stroke performance.
         * // input is an Input.StrokeInput instance
         * input.onStrokeEnd.add(function (event) {
         *     var gesture = deictic.recognisedGesture(deictic.eval(event.d.buffer), 0.70);
         *
         * });
         */
        this.recognizedGesture = function (result, threshold) {
            if (result == null) {
                return null;
            }
            var recognized = null;
            var max = 0.0;
            for (var i in result) {
                var gesture = result[i];
                if (gesture.parts[gesture.parts.length - 1].probability > max) {
                    max = gesture.parts[gesture.parts.length - 1].probability;
                    recognized = gesture.name;
                }
            }

            if (max >= threshold) {
                return recognized;
            } else {
                return null;
            }
        };
    };

    Input.Deictic = Deictic;


    /**
     * @class MachineLearning
     * @memberOf Input
     * @classdesc A support for gesture recognition using Machine Learning (ML) techniques. It allows
     * providing labelled examples of the different gestures in the vocabulary and automatically learns
     * to recognize them.
     * <br/>
     * This object provides a simplified API for the 1$ gesture recogniser described in
     * Jacob O. Wobbrock, Andrew D. Wilson, and Yang Li. 2007. Gestures without libraries, toolkits or training:
     * a $1 recognizer for user interface prototypes. In Proceedings of UIST 2007, ACM, 159-168.
     * {@link https://doi.org/10.1145/1294211.1294238 | DOI}. The javascript library is available at
     * {@link http://depts.washington.edu/madlab/proj/dollar/}
     * <br/>
     * It requires a set of labelled examples represented as an {@link Input~DollarSamples} object, contained in the
     * global variable samples. <br/>
     * For the user-test, a <a href="http://localhost:8000/basic/training" target="_blank">
     *     simple interface for providing the labelled examples</a> is available, please read the
     * tutorial linked below.
     * @description Creates an object for recognizing gestures with the Machine Learning approach.
     * @see Input~DollarTraining
     * @tutorial Recognizing gestures with Machine Learning
     */
    var MachineLearning = function () {
        var _self = this;

        /**
         * @public
         * @instance
         * @method init
         * @description Inits the {@link Input.MachineLearning} instance with the training samples.
         * <br/> Please note that, for library code compatibility issues, the training samples <strong>must
         * be specified through a global variable called samples</strong>.
         * @memberOf Input.MachineLearning
         * @example
         * samples =  ...; // get the labeled samples from any source you like (global)
         * // after this statement, the training samples are not loaded
         * var machineLearning = new Input.MachineLearning();
         * // after this statement the MachineLearning object is ready to use.
         * // The called function  reads internally the samples variable.
         * machineLearning.init()
         */
        this.init = function () {
            _self.r = new DollarRecognizer();
            /**
             * @description Labelled samples for the 1$ gesture recognizer training.
             * @typedef  {Object} Input~DollarTraining
             * @property {string} Name - the name of the gesture (the label)
             * @property {Array<Input~DollarSample>} Points - the points defining the labelled sample
             */
            /**
             * @description Labelled samples for the 1$ gesture recognizer training.
             * @typedef  {Input~DollarSample} Input~DollarSamples
             * @property {number} X - the X coordinate of the sample point
             * @property {number} Y - the Y coordinate of the sample point
             */
            _self.r.Unistrokes = samples;
        };

        /**
         * @public
         * @instance
         * @method eval
         * @description Evaluates the current stroke and returns the probability associated to each gesture label
         * @memberOf Input.MachineLearning
         * @param {Array<Input.Point2D>} sequence - the current stroke represented as an array of {@link Input.Point2D}
         * @returns {Array<Input~MLEvaluation>} the evaluation result.
         * @example
         *
         * // Example 2: stroke update
         * // When receiving an update from the StrokeInput input,
         * // we evaluate it with Machine Learning. The object input is an
         * // Input.StrokeInput instance.
         * var machineLearning = new MachineLearning();
         * machineLearning.init();
         *
         * input.onStrokeChange.add(function (event) {
         *     var result = machineLearning.eval(event.d.buffer);
         *
         *     // check the probability of the V gesture
         *     var prob = 0;
         *     for(var i in result){
         *          var gesture = result[i];
         *          if(gesture.name === 'V'){
         *              prob = gesture.probability;
         *          }
         *     }
         * });
         *
         */
        this.eval = function (sequence) {
            var dollar = _self.r.Recognize(sequence, false);
            var result = [];
            /**
             * @description Probability that the current stroke belongs to the specified named class (label)
             * @typedef  {Object} Input~MLEvaluation
             * @property {string} name - the label (name of the class) for a gesture
             * @property {number} probability - the associated probability
             */
            for (var i in dollar.Rank) {
                result.push({'name': dollar.Rank[i].Name, 'probability': dollar.Rank[i].Prob});
            }

            return result;
        };

        /**
         * @public
         * @instance
         * @method recognisedGesture
         * @description A short-hand for getting the name of the gesture with the higher recognition probability.
         * @memberOf Input.MachineLearning
         * @param {Array<Input~MLEvaluation>} result - the Machine Learning evaluation of a stroke, as returned by the
         * {@link Input.MachineLearning#eval/eval} method.
         * @param {number} threshold - the minimum probability for considering the gesture as recognized.
         * @returns {string} the name of the gesture with the higher evaluation probability, null if all
         * gestures are below the threshold.
         * @example
         * // get the most likely gesture at the end of the stroke performance.
         * // input is an Input.StrokeInput instance
         * input.onStrokeEnd.add(function (event) {
         *     var gesture = machineLearning.recognisedGesture(machineLearning.eval(event.d.buffer), 0.70);
         *
         * });
         */
        this.recognizedGesture = function (result, threshold) {
            var recognized = null;
            var max = 0.0;
            for (var i in result) {
                if (result[i].probability >= max) {
                    recognized = result[i].name;
                    max = result[i].probability;
                }
            }
            if (max >= threshold) {
                return recognized;
            } else {
                return null;
            }
        };
    };

    Input.MachineLearning = MachineLearning;


}(window.Input = window.Input || {}, undefined));