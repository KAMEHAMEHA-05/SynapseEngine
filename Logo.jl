module Logo

using Printf

function clear_screen()
    print("\e[2J\e[H")
end

function move_cursor(line, col)
    print("\e[$(line);$(col)H")
end

# Create smooth gradient color effect
function gradient_color(frame, total_frames=150)
    # Define color keyframes (R, G, B)
    colors = [
        (255, 165, 0),    # Orange
        (0, 150, 255),    # Blue
        (0, 255, 150),    # Green
        (150, 0, 255),    # Purple
        (255, 165, 0)     # Back to Orange for smooth loop
    ]
    
    # Slow down the transition
    frame = frame % total_frames
    segment_length = total_frames ÷ (length(colors) - 1)
    color_index = frame ÷ segment_length
    color_progress = (frame % segment_length) / segment_length
    
    # Interpolate between colors
    c1 = colors[color_index + 1]
    c2 = colors[min(color_index + 2, length(colors))]
    
    r = round(Int, c1[1] + (c2[1] - c1[1]) * color_progress)
    g = round(Int, c1[2] + (c2[2] - c1[2]) * color_progress)
    b = round(Int, c1[3] + (c2[3] - c1[3]) * color_progress)
    
    return "\e[38;2;$(r);$(g);$(b)m"
end

function display_loading()
    frames = ["◐", "◓", "◑", "◒"]
    return frames
end

function display_logo()
    bold = "\e[1m"
    reset = "\e[0m"
    bright = "\e[1m"
    dim = "\e[2m"
    
    logo = """
    ░██████╗██╗░░░██╗███╗░░██╗░█████╗░██████╗░░██████╗███████╗
    ██╔════╝╚██╗░██╔╝████╗░██║██╔══██╗██╔══██╗██╔════╝██╔════╝
    ╚█████╗░░╚████╔╝░██╔██╗██║███████║██████╔╝╚█████╗░█████╗░░
    ░╚═══██╗░░╚██╔╝░░██║╚████║██╔══██║██╔═══╝░░╚═══██╗██╔══╝░░
    ██████╔╝░░░██║░░░██║░╚███║██║░░██║██║░░░░░██████╔╝███████╗
    ╚═════╝░░░░╚═╝░░░╚═╝░░╚══╝╚═╝░░╚═╝╚═╝░░░░░╚═════╝░╚══════╝
        
    ███████╗███╗░░██╗░██████╗░██╗███╗░░██╗███████╗
    ██╔════╝████╗░██║██╔════╝░██║████╗░██║██╔════╝
    █████╗░░██╔██╗██║██║░░██╗░██║██╔██╗██║█████╗░░
    ██╔══╝░░██║╚████║██║░░╚██╗██║██║╚████║██╔══╝░░
    ███████╗██║░╚███║╚██████╔╝██║██║░╚███║███████╗
    ╚══════╝╚═╝░░╚══╝░╚═════╝░╚═╝╚═╝░░╚══╝╚══════╝
    """

    tagline = "⚡ Neural Networks. Reimagined. ⚡"
    
    clear_screen()
    
    lines = split(logo, "\n")
    
    # Calculate center position
    term_width = displaysize(stdout)[2]
    logo_width = maximum(length.(lines))
    start_col = div(term_width - logo_width, 2)
    start_line = 3
    
    # Animation loop with slower transition
    frames = display_loading()
    total_frames = 300  # Increased number of frames for slower transition
    
    for frame in 1:total_frames
        move_cursor(start_line, start_col)
        
        # Get gradient color for current frame
        color = gradient_color(frame, total_frames)
        
        for (i, line) in enumerate(lines)
            move_cursor(start_line + i - 1, start_col)
            print(bold, color, line, reset)
        end
        
        # Display centered tagline
        tagline_col = div(term_width - length(tagline), 2)
        move_cursor(start_line + length(lines) + 1, tagline_col)
        print(bold, color, tagline, reset)
        
        # Display loading animation
        loading_frame = frames[frame % length(frames) + 1]
        loading_text = " Initializing Neural Core "
        loading_col = div(term_width - length(loading_text), 2)
        move_cursor(start_line + length(lines) + 3, loading_col)
        print(dim, loading_frame, loading_text, loading_frame, reset)
        
        sleep(0.1)  # Increased sleep time for slower animation
    end
    
    move_cursor(start_line + length(lines) + 5, 1)
    println()
end

end # module