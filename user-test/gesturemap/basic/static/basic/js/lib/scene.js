/**
 * @namespace Grid
 * @description contains the classes for managing the grid map and the action feedback.
 */
(function (Grid, undefined) {

    BOX_SIZE = 200;


    /**
     * @class Scene
     * @memberOf Grid
     * @description Creates a grid map object
     * @classdesc Represents the grid map to be filled with monsters and treasures.
     */
    var Scene = function () {

        var _self = this;

        /**
         * @public
         * @instance
         * @method init
         * @memberOf Grid.Scene
         * @description Inits the map and the drawing resources
         */
        this.init = function () {
            _self.width = 800;
            _self.height = 600;

            _self.stage = new Konva.Stage({
                container: 'container',
                width: _self.width,
                height: _self.height
            });

            /**
             * @member layer
             * @memberOf Grid.Scene
             * @instance
             * @description the layer ({@link https://konvajs.github.io/docs/ | Konva.Layer}) for drawing the user feedback
             * @type {Konva.Layer}
             */
            _self.layer = new Konva.Layer();
            /**
             * @member imgLayer
             * @memberOf Grid.Scene
             * @instance
             * @description the layer ({@link https://konvajs.github.io/docs/ | Konva.Layer}) for drawing the cell images
             * @type {Konva.Layer}
             */
            _self.imgLayer = new Konva.Layer();
            _self.stage.add(_self.imgLayer);
            _self.stage.add(_self.layer);

            _self.BOX_SIZE = BOX_SIZE;
            var box;
            /**
             * @member state
             * @instance
             * @memberOf Grid.Scene
             * @description a matrix containing the state of the grid map
             * @type {Array}
             */
            _self.state = [];
            // generate boxes
            for (var iy = 0; iy < _self.height / _self.BOX_SIZE; iy++) {
                _self.state[iy] = new Array(Math.trunc(_self.width / _self.BOX_SIZE));
                for (var ix = 0; ix < _self.width / _self.BOX_SIZE; ix++) {
                    box = new Konva.Rect({
                        x: ix * _self.BOX_SIZE,
                        y: iy * _self.BOX_SIZE,
                        width: _self.BOX_SIZE - 1,
                        height: _self.BOX_SIZE - 1,
                        stroke: 'black'
                    });
                    _self.state[iy][ix] = null;

                    _self.layer.add(box);
                }
            }

            _self._monsterImg = new Image();
            _self._monsterImg.src = '/static/basic/img/monster.png';

            _self._treasureImg = new Image();
            _self._treasureImg.src = '/static/basic/img/treasure.png';


            //this.fitStageIntoParentContainer();
        };


        this.fitStageIntoParentContainer = function () {
            var container = document.querySelector('#stage-parent');

            // now we need to fit stage into parent
            var containerWidth = container.offsetWidth;
            // to do this we need to scale the stage
            var scale = containerWidth / _self.width;


            _self.stage.width(_self.width * scale);
            _self.stage.height(_self.height * scale);
            _self.stage.scale({x: scale, y: scale});
            _self.stage.draw();
        }


        /**
         * @public
         * @instance
         * @memberOf Grid.Scene
         * @method setMonster
         * @description Shows a monster icon in the cell that contains a given point P(x,y)
         * @param {number} x the X coordinate of P
         * @param {number} y the Y coordinate of P
         */
        this.setMonster = function (x, y) {
            return this.setImage(
                new Input.Point2D(x, y),
                _self._monsterImg,
                106,
                118);
        };


        /**
         * @public
         * @memberOf Grid.Scene
         * @method setTreasure
         * @instance
         * @description Shows a treasure icon in the cell that contains a given point P(x,y)
         * @param {number} x the X coordinate of P
         * @param {number} y the Y coordinate of P
         */
        this.setTreasure = function (x, y) {
            return this.setImage(
                new Input.Point2D(x, y),
                _self._treasureImg,
                128,
                128);
        };

        /**
         * @private
         * @instance
         * @method setImage
         * @memberOf Grid.Scene
         * @description Shows the specified image in a grid cell
         * @param {Input.Point2D} point a point contained in the cell
         * @param {Image} img the image object to show
         * @param {number} w the image width
         * @param {number} h the image height
         * @see {Input.Point2D}
         */
        this.setImage = function (point, img, w, h) {
            var r = _self.getRow(point);
            var c = _self.getColumn(point);
            var x = (c + 0.5) * _self.BOX_SIZE - 0.5 * w;
            var y = (r + 0.5) * _self.BOX_SIZE - 0.5 * h;

            if (_self.state[r][c] != null) {
                _self.state[r][c].setAttrs(
                    {
                        x: x,
                        y: y,
                        image: img,
                        width: w,
                        height: h
                    }
                );
                _self.state[r][c].show();

            } else {
                // init
                var m = new Konva.Image({
                    x: x,
                    y: y,
                    image: img,
                    width: w,
                    height: h
                });

                // add the shape to the layer
                _self.state[r][c] = m;
                _self.imgLayer.add(m);
            }

            _self.imgLayer.draw();


        };

        /**
         * @public
         * @instance
         * @method clearCell
         * @memberOf Grid.Scene
         * @description Deletes the contents of the cell that contains a given point P(x,y)
         * @param {number} x the X coordinate of P
         * @param {number} y the Y coordinate of P
         */
        this.clearCell = function (x, y) {
            var point = new Input.Point2D(x, y);
            var r = _self.getRow(point);
            var c = _self.getColumn(point);

            if (_self.state[r][c] != null) {
                _self.state[r][c].hide();
                _self.imgLayer.draw();
            }
        };

        this.getColumn = function (point) {
            return Math.trunc(point.x / _self.BOX_SIZE);
        };

        this.getRow = function (point) {
            return Math.trunc(point.y / _self.BOX_SIZE);
        };

    };
    Grid.Scene = Scene;


    /**
     * @class LineFeedback
     * @classdesc A polyline that shows the trajectory of a user's stroke.
     * @memberOf Grid
     * @description Creates a line feedback for a {@link Input.StrokeInput}
     * @see Input.StrokeInput
     */
    var LineFeedback = function () {

        var _self = this;


        /**
         * @public
         * @instance
         * @method init
         * @memberOf Grid.LineFeedback
         * @description Inits the line feedback
         * @param {Konva.Layer} layer the layer for drawing the feedback. See {@link Grid.Scene#layer}
         * @see Grid.Scene
         */
        this.init = function (layer) {
            /**
             * @member layer
             * @memberOf Grid.LineFeedback
             * @instance
             * @description the layer  ({@link https://konvajs.github.io/docs/ | Konva.Layer}) for drawing the user feedback
             * @type {Konva.Layer}
             */
            _self.layer = layer;
            /**
             * @member line
             * @memberOf Grid.LineFeedback
             * @instance
             * @description the line ({@link https://konvajs.github.io/docs/ | Konva.Line}) representing the feedback
             * @type {Konva.Line}
             *
             */
            _self.line = new Konva.Line({
                points: [],
                stroke: 'red',
                strokeWidth: 10,
                lineCap: 'round',
                lineJoin: 'round',

            });

            _self.layer.add(_self.line);
            _self.layer.draw();
        };

        /**
         * @public
         * @instance
         * @method clearLine
         * @memberOf Grid.LineFeedback
         * @description Clears the line feedback
         */
        this.clearLine = function () {
            _self.line.points().splice(0, _self.line.points().length);
            _self.layer.draw();
        };

        /**
         * @public
         * @instance
         * @method addPoint
         * @memberOf Grid.LineFeedback
         * @description Adds a point to the line feedback. The feedback will update its drawing adding a segment between
         * the end of the feedback polyline and the point described by the function parameters.
         * @param {number} x the X coordinate of the point
         * @param {number} y the Y coordinate of the point
         */
        this.addPoint = function (x, y) {
            _self.line.points().push(x, y);
            _self.line.draw();
        };
    };
    Grid.LineFeedback = LineFeedback;

    /**
     * @class Octopocus
     * @classdesc A combined feedback and feedforward component for the user's strokes, as sensed by {@link Input.StrokeInput}.
     * During the stroke execution, it shows a black line
     * representing the part of the trajectory that has been completed (feedback) and the possible completions for the
     * stroke (feedforward). Each possible completion is represented with a different colour, and its opacity represents
     * the estimated completion likelihood.
     * </br> The complete idea has been described in <em>Olivier Bau and Wendy E. Mackay. 2008. OctoPocus: a dynamic
     * guide for learning gesture-based command sets. In Proceedings of UIST '08, ACM,  37-46.</em>
     * {@link https://doi.org/10.1145/1449715.1449724|DOI}</em>
     * @memberOf Grid
     * @description Creates a OctoPocus feedback for a {@link Input.StrokeInput}
     * @see Input.StrokeInput
     */
    var Octopocus = function () {
        var _self = this;

        /**
         * @public
         * @instance
         * @method init
         * @memberOf Grid.Octopocus
         * @description Inits the Octopocus feedback. The completions for the gestures in the user tests (triangle,
         * square and delete) are already set-up.
         * @param {Konva.Layer} layer the layer ({@link https://konvajs.github.io/docs/ | Konva.Layer})
         * for drawing the feedback. See {@link Grid.Scene#layer}
         * @see Grid.Scene
         */
        this.init = function (layer) {
            _self.t = BOX_SIZE - 0.26 * BOX_SIZE;
            _self.r = BOX_SIZE - 0.35 * BOX_SIZE;
            _self.d = BOX_SIZE - 0.35 * BOX_SIZE;

            /**
             * @description Represents the feedforward for the ideal execution of a gesture.
             * @typedef {Object} Grid~Feedforward
             * @property {Konva.Line} line - the line ({@link https://konvajs.github.io/docs/ | Konva.Line}) representing
             * the feedback.
             * @property {Array<Input.Point2D>} points - the points of the polyline representing the ideal gesture
             * execution.
             * @property {number} parts - the number of sub-parts in the ideal execution of the gesture.
             * @property {string} name - the name identifying the specific gesture. For the user test, they are
             * triangle, square and delete.
             * @example
             * {
             *   line: new Konva.Line({
             *       points: [],
             *       stroke: '#008000',
             *       strokeWidth: 6,
             *       lineCap: 'round',
             *       lineJoin: 'round'
             *   }),
             *   points: [
             *       {x: 0, y: 100},
             *       {x: 100, y: 100},
             *       {x: 100, y: 0},
             *       {x: 0, y: 0}
             *   ],
             *   parts: 4,
             *   name: 'square'
             * };
             */
            /**
             * @member feedforward
             * @memberOf Grid.Octopocus
             * @instance
             * @description the description of the possible feedforward
             * @type {Array<Grid~Feedforward>}
             *
             */
            _self.feedforward = [];
            /**
             * @member layer
             * @memberOf Grid.Octopocus
             * @instance
             * @description the layer ({@link https://konvajs.github.io/docs/ | Konva.Layer}) for drawing the user feedback
             * @type {Konva.Layer}
             *
             */
            _self.layer = layer;
            _self.feedforward[0] = {
                line: new Konva.Line({
                    points: [],
                    stroke: '#FF0000',
                    strokeWidth: 6,
                    lineCap: 'round',
                    lineJoin: 'round'
                }),
                points: [
                    {x: -Math.cos(Math.PI / 3) * _self.t, y: Math.sin(Math.PI / 3) * _self.t},
                    {x: Math.cos(Math.PI / 3) * _self.t, y: Math.sin(Math.PI / 3) * _self.t},
                    {x: 0, y: 0}
                ],
                parts: 3,
                name: 'triangle'
            };


            _self.feedforward[1] = {
                line: new Konva.Line({
                    points: [],
                    stroke: '#008000',
                    strokeWidth: 6,
                    lineCap: 'round',
                    lineJoin: 'round'
                }),
                points: [
                    {x: 0, y: _self.r},
                    {x: _self.r, y: _self.r},
                    {x: _self.r, y: 0},
                    {x: 0, y: 0}
                ],
                parts: 4,
                name: 'square'
            };

            _self.feedforward[2] = {
                line: new Konva.Line({
                    points: [],
                    stroke: '#000080',
                    strokeWidth: 6,
                    lineCap: 'round',
                    lineJoin: 'round'
                }),
                points: [
                    {x: _self.d, y: _self.d},
                    {x: 0, y: _self.d},
                    {x: _self.d, y: 0}
                ],
                parts: 3,
                name: 'delete'
            };

            /**
             * @member feedback
             * @memberOf Grid.Octopocus
             * @instance
             * @description the line ({@link https://konvajs.github.io/docs/ | Konva.Line}) representing the actual
             * user's execution of the gesture
             * @type {Konva.Line}
             *
             */
            _self.feedback = new Konva.Line({
                points: [],
                stroke: '#000000',
                strokeWidth: 4,
                lineCap: 'round',
                lineJoin: 'round'
            });

            for (var i = 0; i < _self.feedforward.length; i++) {
                _self.layer.add(_self.feedforward[i].line);
            }
            _self.layer.add(_self.feedback);

            _self.layer.draw();
        };

        /**
         * @public
         * @instance
         * @method clear
         * @memberOf Grid.Octopocus
         * @description Clears both the feedback and the feedforward drawings.
         */
        this.clear = function () {
            for (var i = 0; i < _self.feedforward.length; i++) {
                _self.feedforward[i].line.points().splice(0, _self.feedforward[i].line.points().length);
            }
            _self.feedback.points().splice(0, _self.feedback.points().length);
            _self.layer.draw();
        };

        /**
         * @public
         * @instance
         * @method start
         * @memberOf Grid.Octopocus
         * @description Starts the visualization of both feedback and feedforward.
         * @param {object} event - the original {@link https://konvajs.github.io/docs/ |Konva.Event} enhanced with a
         * {@link Input~StrokeEvent} in the d field
         * @example
         * // It receives the start stroke event from a {@link Input.StrokeInput} instance named
         * // input and  it passes it to a {@link Scene.Octopocus} instance named feedback.
         * // The {@link Input.StrokeInput} instance enhances the {@link https://konvajs.github.io/docs/ |Konva.Event}
         * // descriptor with the d field automatically.
         * input.onStrokeBegin.add(function (event) {
         *     feedback.start(event);
         * });
         */
        /**
         * Represents an update for the feedforward representation.
         * @typedef {Object} Grid~FeedforwardState
         * @property {string} name - the name of the gesture to update
         * @property {number} part - most likely part that the user is currently completing
         * @property {number} probability - the confidence associated by the recognition algorithm to the gesture part
         * @example
         * // we are 66% confident that the user is currently performing the third
         * // side of the square. The part counting starts from 0.
         * var squareUpdate = {name: 'square', part: 2, probability: 0.66}
         */
        this.start = function (event) {
            // punto da cui far partire il feedforward
            _self.p0 = {
                x: event.d.bX + event.d.x,
                y: event.d.bY + event.d.y
            };

            this.update(
                event,
                [
                    {name: 'triangle', part: 0, probability: 0.333},
                    {name: 'square', part: 0, probability: 0.333},
                    {name: 'delete', part: 0, probability: 0.333}
                ]);


        };

        /**
         * @public
         * @instance
         * @method update
         * @memberOf Grid.Octopocus
         * @description Updates the representation of the feedforward.
         * @param {object} event - the original {@link https://konvajs.github.io/docs/ | Konva.Event} enhanced with a
         * {@link Input~StrokeEvent} in the d field.
         * @param {Array<Grid~FeedforwardState>} descr - the feedforward state description
         * @example
         * // It receives the updates on a user's stroke from
         * // a {@link Input.StrokeInput} instance named input and  it passes it to
         * // a {@link Scene.Octopocus} instance named feedback.
         * // The {@link Input.StrokeInput} instance enhances the {@link https://konvajs.github.io/docs/ |Konva.Event}
         * // descriptor with the d field automatically.
         * input.onStrokeChange.add(function (event) {
         *
         *     // perform some computation on the current stroke state
         *         ...
         *
         *
         *     // create the feedforward update
         *     var descr = [
         *          // we are 75% confident that the user is drawing
         *          // the second side of the triangle (part 1)
         *         {name: "triangle", part: 1, probability: 0.75},
         *         // we are 33% confident that the user is drawing
         *         // the first side of the square (part 0)
         *         {name: "square", part: 0, probability: 0.33},
         *         // we are 10% confident that the user is drawing
         *         // the first side of the delete (part 0)
         *         {name: "delete", part: 0, probability: 0.10}
         *      ];
         *
         *     // the triangle will be almost opaque, the square semi-transparent, the delete almost transparent
         *     feedback.update(event, descr);
         * });
         */
        this.update = function (event, descr) {
            var p = {
                x: event.d.bX + event.d.x,
                y: event.d.bY + event.d.y
            };

            for (var s in descr) {
                var state = descr[s]
                for (var f in _self.feedforward) {
                    var feedforward = _self.feedforward[f];
                    if (state.name === feedforward.name) {
                        feedforward.line.points().splice(0, feedforward.line.points().length);
                        feedforward.line.points().push(p.x, p.y)
                        for (var i = state.part; i < feedforward.parts; i++) {
                            feedforward.line.points().push(
                                _self.p0.x + feedforward.points[i].x,
                                _self.p0.y + feedforward.points[i].y)
                        }
                        feedforward.line.opacity(Math.exp(2 * state.probability - 2));
                        break;
                    }
                }
            }

            _self.feedback.points().push(p.x, p.y);
            _self.layer.draw();
        };
    };
    Grid.Octopocus = Octopocus;


//fitStageIntoParentContainer();
// adapt the stage on any window resize
//window.addEventListener('resize', fitStageIntoParentContainer);
}(window.Grid = window.Grid || {}, undefined));