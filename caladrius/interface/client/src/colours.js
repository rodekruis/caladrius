export const yellow = "#ffe119";
export const orange = "#f58231";
export const red = "#e6194B";

export function get_prediction_colour(prediction, lower_bound, upper_bound) {
    return prediction < lower_bound
        ? yellow
        : prediction > upper_bound
        ? red
        : orange;
}

export const contrast_color_array = [
    "#3cb44b",
    "#4363d8",
    "#42d4f4",
    "#f032e6",
    "#fabebe",
    "#469990",
    "#e6beff",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#000075",
];
