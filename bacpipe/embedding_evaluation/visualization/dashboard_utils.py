import panel as pn
import matplotlib
import seaborn as sns
import pandas as pd
import datetime
import logging

logger = logging.getLogger("bacpipe")

sns.set_theme(style="whitegrid")

matplotlib.use("agg")


class DashBoardHelper:
            
    def handle_selection(self, event, widget_idx = None):
        """
        Triggered when the user uses the Lasso or Box select tool.
        """
        if not event.new: 
            return
        
        try:
            selected_points = event.new.get('points', [])
            
            if not selected_points:
                logger.info("Selection cleared")
                return

            logger.info(f"Selected {len(selected_points)} points")
            
            # Extract data from the selected points
            points = {}
            for idx, keys in enumerate(
                ['audiofilename', 'start', 'end', 'index', 'label']
                ):
                points[keys] = [p['customdata'][idx] for p in selected_points]
            
            self.spec_plot_obj[widget_idx]._cache_selected_points(points)
            logger.info(f"First 5 files: {points['audiofilename'][:5]}")
            
        except Exception as e:
            logger.info(f"Error handling selection: {e}")


            
    def save_selected_points(self, event, dialogue_panel, widget_idx):            
            if not hasattr(self.spec_plot_obj[widget_idx], 'selected_points'):
                dialogue_panel.visible = True
                dialogue_panel.value = "No points have been selected."
                return 
            
            points = self.spec_plot_obj[widget_idx].selected_points
            df = pd.DataFrame(points)
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = timestamp + '_selected_points.csv'
            
            self._trigger_spec_obj_update[widget_idx]()
            model_name = self.spec_plot_obj[widget_idx].model_name
            save_path = self.path_func(model_name).plot_path
            
            df.to_csv(save_path / file_name)
            
            dialogue_panel.visible = True
            dialogue_panel.value = (
                f"{len(df)} selected points were save to "
                + str(save_path / file_name)
                )
    
    def handle_click(self, event, widget_idx = 0):
        if not event.new: return
        try:
            point_data = event.new['points'][0]
            logger.info(f"DEBUG CLICK: {point_data}")
            
            # this ensures that the sample rate and 
            # input segment length are set specific to the
            # currently used model
            self._trigger_spec_obj_update[widget_idx]()
            
            # Generate the new figure
            # new_fig = self.update_spectrogram(point_data)
            new_fig = (
                self.spec_plot_obj[widget_idx]
                .update_spectrogram(clickData=point_data)
                )
            
            
            self.spectrogram_plot_panel[widget_idx].object = new_fig
            
            if self.spec_plot_obj[widget_idx].bool_autoplay_audio:
                self.spec_plot_obj[widget_idx].play_audio(event=None)
            
            
        except Exception as e:
            logger.info(f"Error handling click: {e}")    
    
    def update_main_plot(self, p_type, plot_func, widget_idx, **kwargs):
        """Update existing plot, handling both Plotly and matplotlib"""
        plots_dict = getattr(self, f"{p_type}_plot")
        
        # Recreate with save button
        new_panel = self.add_save_button(plot_func, **kwargs)
        plots_dict[widget_idx] = new_panel
        
        # Update the actual figure
        if isinstance(new_panel[0], pn.pane.Plotly):
            # Plotly plot - update object directly
            new_panel[0].object = plot_func(**kwargs)
            
            new_panel[0].param.watch(
                lambda x: self.handle_click(x, widget_idx), 'click_data'
                )
            new_panel[0].param.watch(
                lambda x: self.handle_selection(x, widget_idx), 'selected_data'
                )   
        else:
            # Matplotlib plot wrapped in pn.panel with pn.bind
            # It will auto-update through the binding
            pass
        
        return plots_dict[widget_idx]
            
            
    def init_plot(self, p_type, plot_func, widget_idx, **kwargs):
        getattr(self, f"{p_type}_plot")[widget_idx] = pn.panel(
            self.plot_widget(plot_func, widget_idx=widget_idx, **kwargs), tight=False
        )
        return getattr(self, f"{p_type}_plot")[widget_idx]

    def plot_widget(self, plot_func, **kwargs):
        if kwargs.get("return_fig", False):
            return pn.bind(plot_func, **kwargs)
        else:
            return self.add_save_button(plot_func, **kwargs)

    def widget(self, name, options, attr="Select", width=120, **kwargs):
        return getattr(pn.widgets, attr)(
            name=name, options=options, width=self.widget_width, **kwargs
        )

    def init_widget(self, idx, w_type, **kwargs):
        getattr(self, f"{w_type}_select")[idx] = self.widget(**kwargs)
        return getattr(self, f"{w_type}_select")[idx]
    
    def change_input_options(self, clfier_selection, widget_idx):
        if clfier_selection.new == 'Linear':
            self.btn_run_clfier[widget_idx].name = 'Apply linear classifier'
            self.clfier_path[widget_idx].visible = True
        else:
            self.btn_run_clfier[widget_idx].name = 'Load predictions from integrated classifier'
            self.clfier_path[widget_idx].visible = False

    def add_save_button(self, plot_func, **kwargs):
        """Adds a save button to the plot panel"""
        
        # Check if this is for a Plotly plot by checking if any widgets are passed
        has_widgets = any(hasattr(v, 'value') for v in kwargs.values())
        
        if has_widgets:
            # Create bound figure panel (will auto-update)
            fig_panel = pn.panel(pn.bind(plot_func, **kwargs))
        else:
            # No widgets, just call the function once
            fig_panel = pn.panel(plot_func(**kwargs))

        def save_figure(event):
            # Extract values from widgets
            plot_kwargs = {}
            for key, value in kwargs.items():
                if hasattr(value, "value"):
                    plot_kwargs[key] = value.value
                else:
                    plot_kwargs[key] = value

            # Generate the figure
            fig = plot_func(**plot_kwargs)

            # Generate filename
            if "model_name" in plot_kwargs:
                model_name = plot_kwargs["model_name"]
            elif 'model' in plot_kwargs:
                model_name = plot_kwargs['model']
            else:
                model_name = "all_models"

            plot_type = plot_func.__name__.replace("plot_", "")
            
            if 'predictions_loader' in plot_kwargs:
                label_part = f"{plot_kwargs.get('species', 'unknown')}_{plot_kwargs.get('accumulate_by', 'unknown')}"
            elif 'label_by' in plot_kwargs:
                label_part = plot_kwargs['label_by']
            else:
                label_part = 'plot'
                
            default_filename = f"{model_name}_{plot_type}_{label_part}.png"

            # Determine save path
            if model_name == "all_models":
                save_dir = self.path_func(model_name).plot_path.parent.parent / "overview"
            else:
                save_dir = self.path_func(model_name).plot_path
            save_dir.mkdir(exist_ok=True, parents=True)
            save_path = save_dir / default_filename

            # Save the figure (handle both Plotly and matplotlib)
            try:
                import plotly.graph_objs as go
                if isinstance(fig, go.Figure):
                    fig.write_image(save_path, width=1200, height=800)
                else:
                    fig.savefig(save_path, dpi=300, bbox_inches="tight")
            except Exception as e:
                logger.error(f"Error saving figure: {e}")
                notification.object = f"✗ Error saving: {e}"
                return

            notification.object = f"✓ Figure saved to: {save_path}"

        # Create button and notification
        button = pn.widgets.Button(name="Save Figure", button_type="primary")
        button.on_click(save_figure)
        notification = pn.pane.Markdown("")

        return pn.Column(fig_panel, pn.Row(button), notification)
    
    # def add_save_button(self, plot_func, **kwargs):
    #     """Adds a save button that triggers Plotly's native camera"""
    #     has_widgets = any(hasattr(v, 'value') for v in kwargs.values())
        
    #     if has_widgets:
    #         fig_panel = pn.panel(pn.bind(plot_func, **kwargs))
    #     else:
    #         fig_panel = pn.panel(plot_func(**kwargs))
        
    #     # JavaScript to trigger camera click
    #     trigger_camera_js = """
    #     <script>
    #     function triggerCamera() {
    #         // Find the camera button in the modebar
    #         const cameraBtn = document.querySelector('[data-title*="Download plot"]');
    #         if (cameraBtn) {
    #             cameraBtn.click();
    #         }
    #     }
    #     </script>
    #     <button onclick="triggerCamera()" class="bk bk-btn bk-btn-primary">Save Figure</button>
    #     """
        
    #     custom_button = pn.pane.HTML(trigger_camera_js)
        
    #     # Configure the camera button
    #     def get_filename():
    #         # ... your filename logic ...
    #         print('worked!')
    #         pass
        
    #     config = {
    #         'toImageButtonOptions': {
    #             'format': 'png',
    #             'filename': get_filename(),
    #             'height': 1200,
    #             'width': 1600,
    #             'scale': 2
    #         },
    #         'displaylogo': False
    #     }
        
    #     if isinstance(fig_panel, pn.pane.Plotly):
    #         fig_panel.config = config
        
    #     return pn.Column(fig_panel, custom_button)