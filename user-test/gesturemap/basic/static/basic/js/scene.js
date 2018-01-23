(function (Grid, undefined) {

    BOX_SIZE = 200;

    var Scene = function () {

        var _self = this;

        this.init = function () {
            _self.width = 800;
            _self.height = 600;

            _self.stage = new Konva.Stage({
                container: 'container',
                width: _self.width,
                height: _self.height
            });

            _self.layer = new Konva.Layer();
            _self.imgLayer = new Konva.Layer();
            _self.stage.add(_self.imgLayer);
            _self.stage.add(_self.layer);

            _self.BOX_SIZE = BOX_SIZE;
            var box;
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


        this.setMonster = function (x, y) {
            return this.setImage(
                new Utils.Point2D(x, y),
                _self._monsterImg,
                106,
                118);
        };

        this.setTreasure = function (x, y) {
            return this.setImage(
                new Utils.Point2D(x, y),
                _self._treasureImg,
                128,
                128);
        };

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

        this.clearCell = function (x, y) {
            var point = new Utils.Point2D(x, y);
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

    var LineFeedback = function () {

        var _self = this;

        this.init = function (layer) {
            _self.layer = layer;
            _self.line = new Konva.Line({
                points: [],
                stroke: 'red',
                strokeWidth: 10,
                lineCap: 'round',
                lineJoin: 'round',

            });

            _self.layer.add(_self.line);
            _self.layer.draw();
        }

        this.clearLine = function () {
            _self.line.points().splice(0, _self.line.points().length);
            _self.layer.draw();
        };

        this.addPoint = function (x, y) {
            _self.line.points().push(x, y);
            _self.line.draw();
        };
    };
    Grid.LineFeedback = LineFeedback;

    var Octopocus = function () {
        var _self = this;

        this.init = function (layer) {
            _self.t = BOX_SIZE - 0.35 * BOX_SIZE;
            _self.r = BOX_SIZE - 0.30 * BOX_SIZE;
            _self.d = BOX_SIZE - 0.25 * BOX_SIZE;
            _self.feedforward = [];
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

        this.clear = function () {
            for (var i = 0; i < _self.feedforward.length; i++) {
                _self.feedforward[i].line.points().splice(0, _self.feedforward[i].line.points().length);
            }
            _self.feedback.points().splice(0, _self.feedback.points().length);
            _self.layer.draw();
        };

        this.start = function (event) {
            // punto da cui far partire il feedforward
            _self.p0 = {
                x: event.d.bX + event.d.x,
                y: event.d.bY + event.d.y
            }

            this.update(
                event,
                [
                    {name: 'triangle', part: 0, probability: 0.333},
                    {name: 'square', part: 0, probability: 0.333},
                    {name: 'delete', part: 0, probability: 0.333}
                ]);


        };

        this.update = function (event, states) {
            var p = {
                x: event.d.bX + event.d.x,
                y: event.d.bY + event.d.y
            };

            for (var s in states) {
                var state = states[s]
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
                        feedforward.line.opacity(state.probability);
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