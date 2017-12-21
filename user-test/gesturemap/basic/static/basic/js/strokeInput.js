(function (Deictic, undefined) {
    var Event = function () {

        /**
         * The event callback list
         */
        this.callback = [];

        /**
         * Adds an handler for this event
         * @param {function} handler the handler to be added
         * @returns {undefined}
         */
        this.add = function (handler) {
            this.callback.push(handler);
        };

        /**
         * Removes an handler for this event
         * @param {function} handler the handler to be removed
         * @returns {undefined}
         */
        this.remove = function (handler) {
            var index = this.callback.indexOf(handler);
            if (index > -1) {
                this.callback.splice(index, 1);
            }
        };

        /**
         * Triggers the current event
         * @param {object} evt the event arguments
         * @returns {undefined}
         */
        this.trigger = function (evt) {
            this.callback.forEach(function (l) {
                l(evt);
            });
        };
    };
    Deictic.Event = Event;

    var StrokeInput = function (layer) {
        this.onStrokeBegin = new Event();
        this.onStrokeChange = new Event();
        this.onStrokeEnd = new Event();

        this._layer = layer;
        this._buffer = [];
        this._recording = false;
        this._box = null;

        var _self = this;
        // attach the input source
        this._layer.on('mousedown touchstart', function (event) {
            _self._recording = true;
            _self._box = new Point2D(event.target.attrs.x, event.target.attrs.y);
            _self._buffer.splice(0, _self._buffer.length);
            var point = boxCoordinates(event);
            _self._buffer.push(boxCoordinates(event));
            event.d = {
                x: point.x,
                y: point.y,
                bX: event.target.attrs.x,
                bY: event.target.attrs.y,
                buffer: _self._buffer
            }
            _self.onStrokeBegin.trigger(event);
        });


        this._layer.on('mousemove touchmove', function (event) {
            if (_self._recording === true) {
                var point = boxCoordinates(event);
                _self._buffer.push(boxCoordinates(event));
                event.d = {
                    x: point.x,
                    y: point.y,
                    bX: event.target.attrs.x,
                    bY: event.target.attrs.y,
                    buffer: _self._buffer
                }
                _self.onStrokeChange.trigger(event);
            }
        });

        this._layer.on('mouseup touchend', function (event) {
            _self._recording = false;
            _self._box = null;
            var point = boxCoordinates(event);
            _self._buffer.push(boxCoordinates(event));
            event.d = {
                x: point.x,
                y: point.y,
                bX: event.target.attrs.x,
                bY: event.target.attrs.y,
                buffer: _self._buffer
            }
            _self.onStrokeEnd.trigger(event);
        });

        var boxCoordinates = function (kevent) {
            var point;
            switch (event.type) {
                case 'touchmove':
                case 'touchend':
                    point = new Point(
                        kevent.evt.changedTouches[0].clientX,
                        kevent.evt.changedTouches[0].clientY
                    )
                    break;
                default:
                    point = new Point2D(kevent.evt.x, kevent.evt.y);
                    break;

            }
            console.log('punto: ' + point.x + ', ' + point.y +
                ' l: ' + _self._buffer.length +
                ' k: ' + kevent.target.attrs.x + ', ' + kevent.target.attrs.y);
            return new Point2D(point.x - _self._box.x, point.y - _self._box.y);
        };


    };
    Deictic.StrokeInput = StrokeInput;

    var Point2D = function (x, y) {
        this.x = x;
        this.y = y;
    };
    Deictic.Point2D = Point2D;

    var LineFeedback = function (layer, line) {
        this._line = line;
        this._layer = layer;

        var _self = this;

        this.clearLine = function () {
            _self._line.points().splice(0, _self._line.points().length);
            _self._layer.draw();
        };

        this.addPoint = function (x, y) {
            _self._line.points().push(x, y);
            _self._line.draw();
        };
    };

    Deictic.LineFeedback = LineFeedback;

}(window.Deictic = window.Deictic || {}, undefined));