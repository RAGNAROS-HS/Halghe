var global = require('./global');

class GameControls {
    constructor() {}

    toggleDarkMode() {
        var LIGHT = '#f2fbff',
            DARK = '#181818';
        var LINELIGHT = '#000000',
            LINEDARK = '#ffffff';

        if (global.backgroundColor === LIGHT) {
            global.backgroundColor = DARK;
            global.lineColor = LINEDARK;
        } else {
            global.backgroundColor = LIGHT;
            global.lineColor = LINELIGHT;
        }
    }

    toggleBorder() {
        global.borderDraw = !global.borderDraw;
    }

    toggleMass() {
        global.toggleMassState = global.toggleMassState === 0 ? 1 : 0;
    }

    toggleContinuity() {
        global.continuity = !global.continuity;
    }

    toggleRoundFood(args) {
        if (args || global.foodSides < 10) {
            global.foodSides = (args && !isNaN(args[0]) && +args[0] >= 3) ? +args[0] : 10;
        } else {
            global.foodSides = 5;
        }
    }
}

module.exports = GameControls;
