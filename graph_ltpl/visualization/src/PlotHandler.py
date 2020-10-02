import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.gridspec as gridspec
import numpy as np
import numpy.matlib as npm

# custom modules
import graph_ltpl

# custom packages
import trajectory_planning_helpers as tph

# TUM Colors
TUM_colors = {
    'TUM_blue': '#3070b3',
    'TUM_blue_dark': '#003359',
    'TUM_blue_medium': '#64A0C8',
    'TUM_blue_light': '#98C6EA',
    'TUM_grey_dark': '#9a9a9a',
    'TUM_orange': '#E37222',
    'TUM_green': '#A2AD00'
}


class PlotHandler(object):
    """
    Class that provides several functions to plot internal and external information of an GraphBase object.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        26.10.2018

    """

    def __init__(self,
                 plot_title: str = "Graph Plot",
                 include_timeline: bool = False) -> None:
        """
        :param plot_title:          string specifying the plot window title
        :param include_timeline:    boolean flag specifying whether only the graph ('False') or the graph bundled with
                                    temporal information should be plotted ('True')

        """

        # define canvas
        self.__fig = plt.figure(plot_title, [13, 9])

        # define axes based on configuration
        if not include_timeline:
            self.__main_ax = plt.gca()
            self.__time_ax = "not used"
            self.__time_rel_ax = "not used"
            self.__time_event_ax = "not used"
        else:
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 4])

            # -- TIME EVENT PLOT (RUN OVERVIEW) ------------------------------------------------------------------------
            self.__time_ax = plt.subplot(gs[0])
            self.__time_ax.set_ylim([-0.5, 3.5])
            self.__time_ax.set_yticks([0, 1, 2, 3])
            self.__time_ax.set_yticklabels(['DATA', 'INFO', 'WARNING', 'CRITICAL'])

            self.__time_ax2 = self.__time_ax.twinx()
            self.__time_ax2.set_title("Run analysis")
            self.__time_ax2.set_xlabel('$t$ in s')
            self.__time_ax2.set_ylabel('$v_x$ in m/s'
                                       '\n'
                                       '$\kappa$*250+25 in 1/m')
            self.__time_ax2.grid()

            # in order to still enable onhover event with twinx
            self.__time_event_ax = self.__time_ax.figure.add_axes(self.__time_ax.get_position(True),
                                                                  sharex=self.__time_ax, sharey=self.__time_ax,
                                                                  frameon=False)
            self.__time_event_ax.xaxis.set_visible(False)
            self.__time_event_ax.yaxis.set_visible(False)

            # -- TIME EVENT PLOT (AT CURRENT TIME STEP) ----------------------------------------------------------------
            self.__time_rel_ax = plt.subplot(gs[1])
            self.__time_rel_ax.set_title("Time step analysis")
            self.__time_rel_ax.set_xlabel('$t$ in s')
            self.__time_rel_ax.set_ylabel('$v_x$ in m/s')
            self.__time_rel_ax.set_xlim([0.0, 10.0])
            self.__time_rel_ax.set_ylim([0.0, 60.0])
            self.__time_rel_ax.grid()
            self.__time_rel_ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
            self.__time_rel_ax.grid(which='major', linestyle='-', linewidth='0.6', color='gray')
            self.__time_rel_ax.minorticks_on()

            self.__time_rel_ax2 = self.__time_rel_ax.twinx()
            self.__time_rel_ax2.set_ylabel(r'$\kappa$ in 1/m'
                                           '\n'
                                           r'$\psi$ in rad/(10$\pi$)')
            self.__time_rel_ax2.set_ylim([-0.1, 0.1])

            red_patch = ptch.Patch(color=TUM_colors['TUM_orange'], label='$v_x$')
            blue_patch = ptch.Patch(color=TUM_colors['TUM_blue'], label=r'$\kappa$')
            green_patch = ptch.Patch(color=TUM_colors['TUM_green'], label=r'$\psi$')
            self.__time_rel_ax2.legend(handles=[red_patch, blue_patch, green_patch])

            # -- MAIN PLOT (MAP OVERVIEW) ------------------------------------------------------------------------------
            self.__main_ax = plt.subplot(gs[2])

        # configure main axis
        self.__main_ax.grid()
        self.__main_ax.set_aspect("equal", "datalim")
        self.__main_ax.set_xlabel("east in m")
        self.__main_ax.set_ylabel("north in m")

        # setup event handler object
        self.__eh = EventHandler(self.__fig,
                                 self.__main_ax,
                                 self.__time_event_ax)

        # containers
        self.__obstacle_handle = dict()
        self.__patch_handle = None
        self.__highlight_path = None
        self.__highlight_paths = dict()
        self.__highlight_pos = dict()
        self.__veh_patch = dict()
        self.__text_display = None
        self.__text_display2 = None
        self.__time_annotation = None
        self.__time_rel_line_handle = None

    def plot_graph_base(self,
                        graph_base: graph_ltpl.data_objects.GraphBase.GraphBase,
                        cost_dep_color: bool = True,
                        plot_edges: bool = True) -> None:
        """
        Plot the major components stored in the graph_base object

        :param graph_base:       reference to the GraphBase object instance holding all graph relevant information
        :param cost_dep_color:   boolean flag, specifying, whether to plot edges with a variable color (depending on
                                 cost) or not (Note: cost dependent plotting is drastically slower)
        :param plot_edges:       boolean flag, specifying, whether the edges should be included in the plot

        """

        # refline
        plt_refline, = plt.plot(graph_base.refline[:, 0],
                                graph_base.refline[:, 1],
                                "k--", linewidth=1.4, label="Refline")

        # track bounds
        # bound1 = graph_base.refline + graph_base.normvec_normalized * graph_base.track_width[:, np.newaxis] / 2
        # bound2 = graph_base.refline - graph_base.normvec_normalized * graph_base.track_width[:, np.newaxis] / 2
        bound1 = graph_base.refline + graph_base.normvec_normalized * np.expand_dims(graph_base.track_width_right, 1)
        bound2 = graph_base.refline - graph_base.normvec_normalized * np.expand_dims(graph_base.track_width_left, 1)

        x = list(bound1[:, 0])
        y = list(bound1[:, 1])
        x.append(None)
        y.append(None)
        x.extend(list(bound2[:, 0]))
        y.extend(list(bound2[:, 1]))
        plt_bounds, = self.__main_ax.plot(x, y, "k-", linewidth=1.4, label="Bounds")

        # norm vecs
        x = []
        y = []
        for i in range(bound1.shape[0]):
            temp = np.vstack((bound1[i], bound2[i]))
            x.extend(temp[:, 0])
            y.extend(temp[:, 1])
            x.append(None)
            y.append(None)
        plt_normals, = plt.plot(x, y, color=TUM_colors['TUM_blue_dark'], linestyle="-", linewidth=0.7, label="Normals")

        # raceline points
        rlpt = graph_base.refline + graph_base.normvec_normalized * graph_base.alpha[:, np.newaxis]
        plt_raceline, = self.__main_ax.plot(rlpt[:, 0], rlpt[:, 1], color=TUM_colors['TUM_blue'], linestyle="-",
                                            linewidth=1.4, label="Raceline")

        # plot state poses
        nodes = graph_base.get_nodes()
        i = 0
        x = []
        y = []
        for node in nodes:
            tph.progressbar.progressbar(i, len(nodes) - 1, prefix="Plotting nodes   ")

            # Try to get node info (if filtered, i.e. online graph, this will fail)
            try:
                node_pos = graph_base.get_node_info(node[0], node[1])[0]
                x.append(node_pos[0])
                y.append(node_pos[1])
            except ValueError:
                pass
            i += 1
        plt_nodes, = self.__main_ax.plot(x, y, "x", color=TUM_colors['TUM_blue'], markersize=3, label="Nodes")

        if plot_edges:
            # plot edges
            edges = graph_base.get_edges()
            i = 0

            if not cost_dep_color:
                x = []
                y = []
                color_spline = TUM_colors['TUM_blue_light']  # (0, 1, 0)

                min_cost = None
                max_cost = None
            else:
                # get maximum and minimum cost in all provided edges
                min_cost = 9999.9
                max_cost = -9999.9
                for edge in edges:
                    try:
                        edge_cost = graph_base.get_edge(edge[0], edge[1], edge[2], edge[3])[2]
                        min_cost = min(min_cost, edge_cost)
                        max_cost = max(max_cost, edge_cost)
                    except ValueError:
                        pass

                color_spline = None
                plt_edges = None

            for edge in edges:
                tph.progressbar.progressbar(i, len(edges) - 1, prefix="Plotting edges   ")

                # Try to get edge (if filtered, i.e. online graph, this will fail)
                try:
                    spline = graph_base.get_edge(edge[0], edge[1], edge[2], edge[3])
                    spline_coords = spline[1][:, 0:2]
                    spline_cost = spline[2]

                    # cost dependent color
                    if cost_dep_color:
                        color_spline = (round(min(1, (spline_cost - min_cost) / (max_cost - min_cost)), 2),
                                        round(max(0, 1 - (spline_cost - min_cost) / (max_cost - min_cost)), 2), 0)
                        self.__main_ax.plot(spline_coords[:, 0], spline_coords[:, 1], "-",
                                            color=color_spline, linewidth=0.7)
                    else:
                        # Faster plot method (but for now, no individual color shading)
                        x.extend(spline_coords[:, 0])
                        x.append(None)
                        y.extend(spline_coords[:, 1])
                        y.append(None)
                except ValueError:
                    pass

                i += 1
                # plt.pause(0.000001) # Live plotting -> caution: slows down drastically!

            plt_edges = None
            if not cost_dep_color:
                plt_edges, = self.__main_ax.plot(x, y, "-", color=color_spline, linewidth=0.7, label="Edges")

        # properties
        leg = self.__main_ax.legend(loc='upper left')
        if plot_edges and not cost_dep_color:
            elements = [plt_refline, plt_bounds, plt_normals, plt_raceline, plt_nodes, plt_edges]
        else:
            elements = [plt_refline, plt_bounds, plt_normals, plt_raceline, plt_nodes]
        elementd = dict()
        # couple legend entry to real line
        for leg_element, orig_element in zip(leg.get_lines(), elements):
            leg_element.set_pickradius(10)  # 5 pts tolerance
            elementd[leg_element] = orig_element

        # line picking
        self.__fig.canvas.mpl_connect('pick_event', lambda event: self.__eh.onpick(event=event,
                                                                                   elementd=elementd))

        # detail information
        node_plot_marker, = self.__main_ax.plot([], [], 'o', color=TUM_colors['TUM_orange'])
        edge_plot_marker, = self.__main_ax.plot([], [], '-', color=TUM_colors['TUM_orange'])
        annotation = self.__main_ax.annotate('', xy=[0, 0], xytext=(0, 0), arrowprops={'arrowstyle': "->"})
        self.__eh.set_graph_markers(node_plot_marker=node_plot_marker,
                                    edge_plot_marker=edge_plot_marker,
                                    annotation=annotation)
        self.__fig.canvas.mpl_connect('motion_notify_event',
                                      lambda event: self.__eh.onhover(event=event,
                                                                      graph_base=graph_base))

        self.__text_display = self.__main_ax.text(0.02, 0.95, "", transform=plt.gcf().transFigure)
        self.__text_display2 = self.__main_ax.text(0.8, 0.9, "", transform=plt.gcf().transFigure)

        if type(self.__time_ax) is not str:
            self.__time_annotation = self.__time_ax.annotate("", xy=(0, 0), xytext=(0.05, 0.90),
                                                             textcoords='figure fraction',
                                                             bbox=dict(boxstyle="round", fc="w"),
                                                             arrowprops=dict(arrowstyle="->"))

    def update_obstacles(self,
                         obstacle_pos_list: list = None,
                         obstacle_radius_list: list = None,
                         object_id: str = 'default',
                         color: str = 'TUM_green') -> None:
        """
        Update the obstacle indicators in the plot (default is 'None' -> obstacles removed).

        :param obstacle_pos_list:       array of x, y poses of circular obstacles to be plotted
        :param obstacle_radius_list:    array of radi in m for each obstacle provided via 'obstacle_pos_list'
        :param object_id:               sting with an unique object ID (previously plotted objects with same ID will be
                                        removed first)
        :param color:                   string specifying color (use own colors specified in config or default strings)

        """

        # translate color string, if TUM color
        if color in TUM_colors.keys():
            color = TUM_colors[color]

        # delete obstacles with stored handle
        if object_id in self.__obstacle_handle.keys():
            for handle in self.__obstacle_handle[object_id]:
                handle.remove()

            # reset obstacle handle list
            del self.__obstacle_handle[object_id]

            # if existing legend handles
            if (object_id + "_legend") in self.__obstacle_handle.keys():
                self.__obstacle_handle[object_id + "_legend"].remove()
                del self.__obstacle_handle[object_id + "_legend"]

        # obstacles
        if obstacle_pos_list is not None:
            # check if an equal amount of radii and positions is provided
            if len(obstacle_pos_list) is not len(obstacle_radius_list):
                raise ValueError(
                    'The inputs "obstacle_pos_list" and "obstacle_radius_list" should hold the same amount of '
                    'entries!')

            # Init object handle variable as list
            self.__obstacle_handle[object_id] = []

            # plot each obstacle
            for obstacle_pos, obstacle_radius in zip(obstacle_pos_list, obstacle_radius_list):
                plt_circle = plt.Circle(obstacle_pos, obstacle_radius, color=color, fill=True, zorder=10)
                handle = self.__main_ax.add_artist(plt_circle)
                self.__obstacle_handle[object_id].append(handle)

            # add further plot in order to generate legend with proper marker
            if obstacle_pos_list:
                self.__obstacle_handle[object_id + "_legend"], = self.__main_ax.plot(obstacle_pos_list[0][0],
                                                                                     obstacle_pos_list[0][1],
                                                                                     color=color, marker="o", ls="",
                                                                                     zorder=0, label=object_id)

            self.__main_ax.legend()

    def highlight_patch(self,
                        patch_xy_pos_list: list) -> None:
        """
        Highlight a one or more patches defined by the bound positions.

        :param patch_xy_pos_list:        list of x, y position numpy-arrays (each a column) defining the outer bounds

        """

        # delete patches with stored handle
        if self.__patch_handle is not None:
            for handle in self.__patch_handle:
                handle.remove()

            # Reset patch handle list
            self.__patch_handle = None

        # obstacles
        if patch_xy_pos_list:
            # Init object handle variable as list
            self.__patch_handle = []

            # plot each patch
            for patch_xy_pos in patch_xy_pos_list:
                plt_patch = ptch.Polygon(patch_xy_pos, facecolor="black", alpha=0.5, zorder=5)
                handle = self.__main_ax.add_artist(plt_patch)
                self.__patch_handle.append(handle)

    def update_text_field(self,
                          text_str: str,
                          color_str: str = 'k',
                          text_field_id: int = 1) -> None:
        """
        Update a text field in the plot window.

        :param text_str:        text string to be displayed
        :param color_str:       string specifying color (use default color strings only)
        :param text_field_id:   int for the text field to be addressed (currently 1 or 2)

        """

        if text_field_id == 1:
            self.__text_display.set_text(text_str)
            self.__text_display.set_color(color_str)
        elif text_field_id == 2:
            self.__text_display2.set_text(text_str)
            self.__text_display2.set_color(color_str)
        else:
            print("No text_field_id '%i' defined!" % text_field_id)

    def highlight_edges(self,
                        graph_base: graph_ltpl.data_objects.GraphBase.GraphBase,
                        passed_nodes: list,
                        color_str='m') -> None:
        """
        Highlight a sequence of edges in the graph plot with another color (specified by node list).

        :param graph_base:           reference to the class object holding all graph relevant information
        :param passed_nodes:         a list of nodes (layer and node id tuple), which define the path to be highlighted
        :param color_str:            character specifying the color of the generated line

        """

        # generate coordinates for highlight edges
        spline_coord_x = []
        spline_coord_y = []
        previous_node = None
        for current_node in passed_nodes:
            if previous_node is not None:
                spline_param = graph_base.get_edge(start_layer=previous_node[0],
                                                   start_node=previous_node[1],
                                                   end_layer=current_node[0],
                                                   end_node=current_node[1])[1]

                # Add the coordinates to a single array with "None" values separating the splines
                spline_coord_x.extend(spline_param[:, 0])  # x
                spline_coord_x.append(None)
                spline_coord_y.extend(spline_param[:, 1])  # y
                spline_coord_y.append(None)

            previous_node = current_node

        # plot the spline
        self.highlight_line(line_coords_x=spline_coord_x,
                            line_coords_y=spline_coord_y,
                            color_str=color_str)

    def highlight_line(self,
                       line_coords_x: list,
                       line_coords_y: list,
                       color_str: str = "r") -> None:
        """
        Highlight a given coordinate sequence.

        :param line_coords_x:        list of the x coordinates to be highlighted
        :param line_coords_y:        list of the y coordinates to be highlighted
        :param color_str:            character specifying the color of the generated line (use default color stings)

        """

        # delete highlighted paths with stored handle
        if self.__highlight_path is not None:
            self.__highlight_path.remove()

        # plot the spline
        self.__highlight_path, = self.__main_ax.plot(line_coords_x, line_coords_y, color_str + "-",
                                                     linewidth=1.4, label="Local Path")

    def highlight_pos(self,
                      pos_coords: list,
                      color_str: str = 'y',
                      zorder: int = 10,
                      radius: float = None,
                      id_in: str = 'default') -> None:
        """
        Highlight a position with a simple circular shape.

        :param pos_coords:  list of x and y coordinates for the objects (first list holding only x, second only y coord)
        :param color_str:   string specifying color (use default color strings)
        :param zorder:      z-order of the plotted object (layer in the plot)
        :param radius:      radius of the object to be plotted (if set to 'None', a standard dot marker is used)
        :param id_in:       sting with an unique object ID (previously plotted objects with same ID will be removed)

        """

        # delete highlighted positions with handle
        if id_in in self.__highlight_pos.keys():
            self.__highlight_pos[id_in].remove()
            del self.__highlight_pos[id_in]

            if (id_in + "_legend") in self.__highlight_pos.keys():
                self.__highlight_pos[id_in + "_legend"].remove()
                del self.__highlight_pos[id_in + "_legend"]

        # plot pos
        if radius is None:
            self.__highlight_pos[id_in], =  self.__main_ax.plot(pos_coords[0], pos_coords[1], marker="o", ls="",
                                                                color=color_str, zorder=zorder, label=id_in)
        else:
            plt_circle = plt.Circle(tuple(pos_coords), radius, color=color_str, fill=True, zorder=zorder, label=id_in)
            self.__highlight_pos[id_in] = self.__main_ax.add_artist(plt_circle)

            # add further plot in order to generate legend with proper marker
            self.__highlight_pos[id_in + "_legend"], = self.__main_ax.plot(pos_coords[0], pos_coords[1], color=color_str
                                                                           , marker="o", ls="", zorder=0, label=id_in)

        self.__main_ax.legend()

    def plot_vehicle(self,
                     pos: np.ndarray,
                     heading: float,
                     width: float,
                     length: float,
                     zorder: int = 10,
                     color_str: str = 'blue',
                     id_in: str = 'default') -> None:
        """
        Highlight a vehicle pose with a scaled bounding box.

        :param pos:         position of the vehicle's center of gravity
        :param heading:     heading of the vehicle (0.0 beeing north) [in rad]
        :param width:       width of the vehicle [in m]
        :param length:      length of the vehicle [in m]
        :param zorder:      z-order of the plotted object (layer in the plot)
        :param color_str:   string specifying color (use default color strings)
        :param id_in:       sting with an unique object ID (previously plotted objects with same ID will be removed)

        """

        # delete highlighted positions with handle
        if id_in in self.__veh_patch.keys():
            self.__veh_patch[id_in].remove()
            del self.__veh_patch[id_in]

        theta = heading - np.pi / 2

        bbox = (npm.repmat([[pos[0]], [pos[1]]], 1, 4)
                + np.matmul([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                            [[-length / 2, length / 2, length / 2, -length / 2],
                             [-width / 2, -width / 2, width / 2, width / 2]]))

        patch = np.array(bbox).transpose()
        patch = np.vstack((patch, patch[0, :]))

        plt_patch = ptch.Polygon(patch, facecolor=color_str, zorder=zorder)
        self.__veh_patch[id_in] = self.__main_ax.add_artist(plt_patch)

    def highlight_lines(self,
                        line_coords_list: list,
                        id_in: str = 'default',
                        color_base: list = [1, 0, 0]) -> None:
        """
        Highlight a list of given coordinate sequences (each with a different color).

        :param line_coords_list:     list of lists holding the coordinates of paths each
        :param id_in:                id used for the handle (plots with the same id will be removed before plotting)
        :param color_base:           basic color to be used to generate spectrum of this color for each path alternative
                                     (only use colors with ones and zeros!)

        """

        # delete highlighted paths with handle
        if id_in in self.__highlight_paths.keys():
            for handle in self.__highlight_paths[id_in]:
                handle.remove()

        # construct one single line (separated by 'None') and generate color map
        self.__highlight_paths[id_in] = []

        # plot lines in array (plot reversed, such that the last option is on the lowest layer in the plot)
        for idx, line_coords in enumerate(reversed(line_coords_list)):
            # generate a color for the spline
            color = (1 - ((1 + idx) / len(line_coords_list)) * (1 - color_base[0]),
                     1 - ((1 + idx) / len(line_coords_list)) * (1 - color_base[1]),
                     1 - ((1 + idx) / len(line_coords_list)) * (1 - color_base[2]))

            # plot the spline
            line_coords = np.array(line_coords)
            x_lim = self.__main_ax.get_xlim()
            y_lim = self.__main_ax.get_ylim()
            temp_handle, = self.__main_ax.plot(line_coords[:, 0], line_coords[:, 1], color=color,
                                               linewidth=1.4, label=id_in if idx == 0 else "", zorder=99)
            self.__main_ax.set_xlim(x_lim)
            self.__main_ax.set_ylim(y_lim)

            # append handle to array
            self.__highlight_paths[id_in].append(temp_handle)

    def plot_timeline_stamps(self,
                             time_stamps: list,
                             types: list,
                             lambda_fct) -> None:
        """
        Plot an interactive time line overview for logging messages stored in a file.

        :param time_stamps:          list of float values holding time in seconds
        :param types:                list of strings indicating the type of the provided timestamp (e.g. 'WARNING')
        :param lambda_fct:           lambda-function to be called, when mouse is moved over time line

        """

        type_names = ["DATA", "INFO", "WARNING", "CRITICAL"]
        type_marker = {"DATA": "gx", "INFO": "bx", "WARNING": "yx", "CRITICAL": "rx"}

        for i, type_name in enumerate(type_names):
            if type_name in types:
                rel_time_stamps = [x for x, t in zip(time_stamps, types) if t == type_name]
                self.__time_ax.plot(rel_time_stamps, np.zeros((len(rel_time_stamps), 1)) + i, type_marker[type_name])

        time_line_marker, = self.__time_ax.plot([], [], 'r-')
        self.__eh.set_time_markers(time_line_marker=time_line_marker,
                                   lambda_fct=lambda_fct)

    def plot_timeline_course(self,
                             line_coords_list: list) -> None:
        """
        Plot an interactive time line overview for one or multiple data course(s) (e.g. velocity) in a log file.

        :param line_coords_list:    list of numpy arrays, each holding time stamps in the first column and data points
                                    in the second

        """

        # color masks
        color = [TUM_colors['TUM_orange'], TUM_colors['TUM_blue'], TUM_colors['TUM_green']]
        group_axes = [self.__time_ax2, self.__time_ax2, self.__time_ax2]

        # plot lines in array (plot reversed, such that the last option is on the lowest layer in the plot)
        for idx, line_coords in enumerate(line_coords_list):
            # plot the spline
            line_coords = np.array(line_coords)
            group_axes[idx].plot(line_coords[:, 0], line_coords[:, 1], color[idx], linewidth=1.4, label="Local Path",
                                 zorder=0)

    def highlight_timeline(self,
                           time_stamp: float,
                           type_in: str,
                           message: str) -> None:
        """
        Highlight a message within the time-line with an annotation arrow and a text blob.

        :param time_stamp:  time-stamp to be highlighted
        :param type_in:     type of message to be highlighted (e.g. 'WARNING')
        :param message:     message text to be displayed in the blob.

        """

        type_names = ["DATA", "INFO", "WARNING", "CRITICAL"]
        type_marker = {"DATA": "g", "INFO": "b", "WARNING": "y", "CRITICAL": "r"}

        pos = [time_stamp, type_names.index(type_in)]
        self.__time_annotation.xy = pos

        self.__time_annotation.set_text(message)
        self.__time_annotation.get_bbox_patch().set_facecolor(type_marker[type_in])
        self.__time_annotation.get_bbox_patch().set_alpha(0.4)

        self.__time_annotation.get_visible()

    def plot_time_rel_line(self,
                           line_coords_list: list) -> None:
        """
        Highlight a list of given coordinate sequences (each with a different color).

        :param line_coords_list:  list of lists (grouped sets) of lists holding the coordinates of paths each

        """

        # color masks
        color_masks = [TUM_colors['TUM_orange'], TUM_colors['TUM_blue'], TUM_colors['TUM_green']]
        group_axes = [self.__time_rel_ax, self.__time_rel_ax2, self.__time_rel_ax2]

        # delete existing time courses with stored handle
        if self.__time_rel_line_handle is not None:
            for handle in self.__time_rel_line_handle:
                handle.remove()

        #
        self.__time_rel_line_handle = []

        # plot lines in array (plot reversed, such that the last option is on the lowest layer in the plot)
        for group_idx, group_data in enumerate(line_coords_list):
            for idx, line_coords in enumerate(reversed(group_data)):
                # generate a color for the line
                fade_in_clr = (idx + 1) / len(group_data)

                # plot the spline
                line_coords = np.array(line_coords)
                temp_handle, = group_axes[group_idx].plot(line_coords[:, 0], line_coords[:, 1],
                                                          color=color_masks[group_idx],
                                                          linewidth=1.4, label="Local Path", zorder=99 + idx,
                                                          alpha=fade_in_clr)
                # append handle to array
                self.__time_rel_line_handle.append(temp_handle)

    @staticmethod
    def show_plot(non_blocking: bool = False) -> None:
        """
        Show plot, either blocking (if no interactive time line) or non-blocking in order to allow interaction.

        :param non_blocking:    boolean flag, specifying, whether to call blocking or non-blocking update

        """

        if non_blocking:
            plt.draw()
            plt.pause(0.0001)
        else:
            plt.show()


class EventHandler:
    """
    Class that provides interactive event handler functions (e.g. actions to be executed when mouse was moved)

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        26.10.2018

    """

    def __init__(self,
                 fig,
                 main_ax,
                 time_ax) -> None:
        """
        :param fig:         handle for the figure
        :param main_ax:     handle for the main axis
        :param time_ax:     handle for the time axis

        """

        self._fig = fig
        self._main_ax = main_ax
        self._time_ax = time_ax

        self._node_plot_marker = None
        self._edge_plot_marker = None
        self._annotation = None
        self._time_marker = None
        self._lambda_fct = None

    def set_graph_markers(self,
                          node_plot_marker,
                          edge_plot_marker,
                          annotation) -> None:
        """
        Store handles for highlight functions in the main plot.

        :param node_plot_marker:    handle for node highlight (all initialized, only data update required)
        :param edge_plot_marker:    handle for edge highlight (all initialized, only data update required)
        :param annotation:          handle for text annotation (all initialized, only data update required)

        """

        self._node_plot_marker = node_plot_marker
        self._edge_plot_marker = edge_plot_marker
        self._annotation = annotation

    def set_time_markers(self,
                         time_line_marker,
                         lambda_fct) -> None:
        """
        Store handles for highlight in time line window.

        :param time_line_marker:    handle for time line higlight (red line - all initialized, only data update req.)
        :param lambda_fct:          lambda function to be called when mouse is moved over time-line

        """

        self._time_marker = time_line_marker
        self._lambda_fct = lambda_fct

    def onpick(self, event, elementd) -> None:
        """
        Process mouse clicks. Here, we catch clicks on legend elements and hide corresponding data plots.

        :param event:       mouse event object
        :param elementd:    clicked legend element

        """

        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        leg_element = event.artist
        orig_element = elementd[leg_element]
        vis = not orig_element.get_visible()
        orig_element.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            leg_element.set_alpha(1.0)
        else:
            leg_element.set_alpha(0.2)
        self._fig.canvas.draw()

    def onhover(self, event, graph_base) -> None:
        """
        Catch mouse hovering over plot and trigger corresponding actions. Here we update the temporal plot and vehicle
        visualizations, when moving the mouse over the time line. Furthermore, we highlight nodes and connecting edges
        when hovering over the spatial graph.

        :param event:       mouse event object
        :param graph_base:  GraphBase object instance (e.g. used to get closest nodes to mouse pointer)

        """

        if event.inaxes == self._main_ax:
            # get closest node for given coordinates
            node, distances = graph_base.get_closest_nodes([event.xdata, event.ydata], 1)

            # if distance smaller than certain threshold
            if distances[0] < 2:
                # get further node details
                pos, psi, raceline, children, _ = graph_base.get_node_info(layer=node[0][0],
                                                                           node_number=node[0][1],
                                                                           return_child=True)

                # highlight node
                self._node_plot_marker.set_data(pos)

                # -- plot info --
                # set marker pos
                self._annotation.xy = pos

                # determine and set text position (relative to current zoom level)
                dx = (self._main_ax.get_xlim()[1] - self._main_ax.get_xlim()[0]) * 0.1
                dy = (self._main_ax.get_ylim()[1] - self._main_ax.get_ylim()[0]) * 0.01
                self._annotation.set_position([pos[0] + dx, pos[1] + dy])

                # highlight children
                spline_coord_x = []
                spline_coord_y = []

                cost_str = ""
                for child in children:
                    _, spline_param, offline_cost, spline_length = \
                        graph_base.get_edge(start_layer=node[0][0],
                                            start_node=node[0][1],
                                            end_layer=child[0],
                                            end_node=child[1])

                    spline_coord_x.extend(spline_param[:, 0])  # x
                    spline_coord_x.append(None)
                    spline_coord_y.extend(spline_param[:, 1])  # y
                    spline_coord_y.append(None)

                    # Generate cost information
                    kappa_avg = np.power(sum(abs(spline_param[:, 3])) / float(len(spline_param[:, 3])), 2)
                    kappa_peak = np.power(abs(max(spline_param[:, 3]) - min(spline_param[:, 3])), 2)
                    cost_str = (cost_str + "[" + str(node[0][1]) + "-" + str(child[1])
                                + "]: %.3f (ĸ_av²: %.3f, ĸ_peak²: %.3f)\n" % (offline_cost, kappa_avg, kappa_peak))

                # set annotation text
                self._annotation.set_text("Layer ID: " + str(node[0][0]) + "\n"
                                          "Node ID: " + str(node[0][1]) + "\n"
                                          "PSI: " + "%.3f" % psi + "\n"
                                          "Cost:\n" + cost_str)

                self._edge_plot_marker.set_data([spline_coord_x, spline_coord_y])

                self._fig.canvas.draw()

        elif event.inaxes == self._time_ax:
            self._time_marker.set_data([event.xdata, event.xdata], self._time_ax.get_ylim())
            self._fig.canvas.draw()

            self._lambda_fct(event.xdata)
