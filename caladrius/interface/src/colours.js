export const least = 'orange'
export const partial = 'purple'
export const heavy = 'steelBlue'
export const selected = "red"

export function get_colour_from_x(x, a, b) {
    return x < a ? least : x > b ? heavy : partial;
}

export function get_point_colour(x, a, b, x_id, selected_id) {
    if (x_id === selected_id) {
        return selected
    }
    else {
        return get_colour_from_x(x, a, b)
    }
}
