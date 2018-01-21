(function (Utils, undefined) {
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
    Utils.Event = Event;

    var StrokeInput = function (layer) {
        this.onStrokeBegin = new Event();
        this.onStrokeChange = new Event();
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
            //console.log('punto: ' + point.x + ', ' + point.y +
            //    ' l: ' + _self._buffer.length +
            //    ' k: ' + kevent.target.attrs.x + ', ' + kevent.target.attrs.y);
            return new Point2D(point.x - _self._box.x, point.y - _self._box.y);
        };

        var createEvent = function (point) {
            return {
                x: point.x,
                y: point.y,
                bX: _self._box.x,
                bY: _self._box.y,
                buffer: _self._buffer
            };
        }


    };
    Utils.StrokeInput = StrokeInput;

    var Point2D = function (x, y) {
        this.x = x;
        this.y = y;
    };
    Utils.Point2D = Point2D;

    var AngleFSM = function () {
        var _self = this;


        this.init = function (stateDesc, tollerance) {
            _self.state = -1;
            _self.error = 0;
            _self.next = 0;
            _self.states = stateDesc;
            _self.tollerance = tollerance != null ? tollerance : 3;
        };

        this.restart = function () {
            _self.state = -1;
            _self.error = 0;
            _self.next = 0;
        };

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

            if (_self.error > _self.tollerance) {
                _self.state = -2;
            }

            if (_self.next > _self.tollerance) {
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
    Utils.AngleFSM = AngleFSM;

    var angle = function (ax, ay) {
        var angle = Math.acos(ax / Math.sqrt(ax * ax + ay * ay));
        if (ay > 0) {
            angle = 2 * Math.PI - angle;
        }
        return angle * 180 / Math.PI;
    };
    Utils.angle = angle;


}(window.Utils = window.Utils || {}, undefined));