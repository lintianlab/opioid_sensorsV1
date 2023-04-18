%%% Stack Pixel-wise dF/F analysis
%%% Nikki Tjahjono
%%% 07/22/21 Updated
classdef timeconffull_exported < matlab.apps.AppBase
    
    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        EdgeDetectionMethodButtonGroup  matlab.ui.container.ButtonGroup
        CannyButton                     matlab.ui.control.RadioButton
        LogButton                       matlab.ui.control.RadioButton
        SobelButton                     matlab.ui.control.RadioButton
        StackNumberEditFieldLabel       matlab.ui.control.Label
        StackNumberEditField            matlab.ui.control.NumericEditField
        ShowMaskButton                  matlab.ui.control.Button
        SelectImageButton               matlab.ui.control.Button
        FudgeFactorEditFieldLabel       matlab.ui.control.Label
        FudgeFactorEditField            matlab.ui.control.NumericEditField
        DilateEditFieldLabel            matlab.ui.control.Label
        DilateEditField                 matlab.ui.control.NumericEditField
        MedianBackgroundSubtractCheckBox  matlab.ui.control.CheckBox
        FeatureRegistrationCheckBox     matlab.ui.control.CheckBox
        GenerateHeatmapandHistogramCheckBox  matlab.ui.control.CheckBox
        MaxDFFEditFieldLabel            matlab.ui.control.Label
        MaxDFFEditField                 matlab.ui.control.NumericEditField
        StartBeforeLigandEditFieldLabel  matlab.ui.control.Label
        StartBeforeLigandEditField      matlab.ui.control.NumericEditField
        InputstacknumbersbeforeligandforaveragingLabel  matlab.ui.control.Label
        UITable                         matlab.ui.control.Table
        StackEditField_2Label           matlab.ui.control.Label
        StackEditField_2                matlab.ui.control.NumericEditField
        OutputTableLabel                matlab.ui.control.Label
        NikkiTjahjonoV12020Label        matlab.ui.control.Label
        RunPixelWiseAnalysisButton      matlab.ui.control.Button
        EndBeforeLigandEditFieldLabel   matlab.ui.control.Label
        EndBeforeLigandEditField        matlab.ui.control.NumericEditField
        MaxSNREditFieldLabel            matlab.ui.control.Label
        MaxSNREditField                 matlab.ui.control.NumericEditField
        UIAxes                          matlab.ui.control.UIAxes
        UIAxes3                         matlab.ui.control.UIAxes
        UIAxes2                         matlab.ui.control.UIAxes
    end

    
    properties (Access = public)
        Stack % Full Image File
        EdgeMethod %Edge Detect Method
        Title
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            addpath("Dependencies/")
        end

        % Button pushed function: SelectImageButton
        function SelectImageButtonPushed(app, event)
            global stack
            global imageloaded
            [file, path] = uigetfile({'*.lsm';},'Select All Image Files')
            app.Title=file;
            data= bfopen(fullfile(path,file));
            stack=data{1,1};
            app.Stack=stack;
            [j k]=size(stack);
            imageloaded=stack{j,1};
            title(app.UIAxes, app.Title);
            title(app.UIAxes3, app.Title);
            xlabel(app.UIAxes, []);
            ylabel(app.UIAxes, []);
            app.UIAxes.XAxis.TickLabels = {};
            app.UIAxes.YAxis.TickLabels = {};
            
            imshow(imageloaded, 'parent', app.UIAxes);
           
            
        end

        % Button pushed function: ShowMaskButton
        function ShowMaskButtonPushed(app, event)
            ff=app.FudgeFactorEditField.Value;
            dil=app.DilateEditField.Value;
            stacknum=app.StackNumberEditField.Value;
            dat=app.Stack{stacknum, 1};
            title(app.UIAxes2, "Mask");
            xlabel(app.UIAxes2, []);
            ylabel(app.UIAxes2, []);
            app.UIAxes.XAxis.TickLabels = {};
            app.UIAxes.YAxis.TickLabels = {};
            switch app.EdgeDetectionMethodButtonGroup.SelectedObject.Text
                case 'Canny' %where "Button1" is the name of the first radio button
                    mask=edgeCannySegv2(dat, ff, dil);
                    imshow(mask,'parent', app.UIAxes2);
                    imshow(dat, 'parent', app.UIAxes);
                    app.EdgeMethod='Canny'
                case 'Log'
                    mask=edgelogSegv2(dat, ff, dil);
                    imshow(mask,'parent', app.UIAxes2);
                    imshow(dat, 'parent', app.UIAxes);
                    app.EdgeMethod='Log'
                case 'Sobel'
                    mask=edgeSobelSegv2(dat, ff, dil);
                    imshow(mask,'parent', app.UIAxes2);
                    imshow(dat, 'parent', app.UIAxes);
                    app.EdgeMethod='Sobel'
                otherwise
            end
        end

        % Button pushed function: RunPixelWiseAnalysisButton
        function RunPixelWiseAnalysisButtonPushed(app, event)
            selpath=uigetdir("~/Documents/");
            framebefore1=app.StartBeforeLigandEditField.Value;
            framebefore2=app.EndBeforeLigandEditField.Value;
            backgroundsub=app.MedianBackgroundSubtractCheckBox.Value;
            register=app.FeatureRegistrationCheckBox.Value;
            [T, dffs, nonzeromask, snrs] = timeconf_guimedian(app.Stack, framebefore1, framebefore2, backgroundsub, ...
                                            register, app.EdgeDetectionMethodButtonGroup.SelectedObject.Text, app.FudgeFactorEditField.Value, app.DilateEditField.Value);
            plot(T{:,2}, '.-', 'parent', app.UIAxes3);
            app.UITable.Data=T;
            writetable(T, strcat(selpath, "/", app.Title(1:end-4), ".csv"));
            if app.GenerateHeatmapandHistogramCheckBox.Value
                
                figure %plots template overlay on averageDFF
                subplot(2,2,1) %changed from subplot(2,1,1)
                title('dF/F Heatmap')
                hold on
                colormap(jet(10));
                imagesc(dffs(:,:,app.StackEditField_2.Value))
                colorbar
                caxis([0 app.MaxDFFEditField.Value])
                axis off
                daspect([1 1 1])
                hold off
                subplot(2,2,[3,4]) %changed from subplot(2,1,2)
                title('Pixel Median DFF')
                hold on
                dffofint=dffs(:,:,app.StackEditField_2.Value)
                first=dffs(:,:,1)%changed from 1 to 10
                ind_nonzero1=nonzeromask{1} %changed from 1 to 10
                ind_nonzeroofint=nonzeromask{app.StackEditField_2.Value}
                h1=histogram(dffofint(ind_nonzeroofint));
                h1.Normalization = 'probability';
                h1.BinWidth = 0.5;
                h2=histogram(first(ind_nonzero1));
                h2.Normalization = 'probability';
                h2.BinWidth = 0.5;
                xlim([-1 30]); %changed from -1 30
                xlabel("DFF");
                ylabel("probability");
                hold off
                subplot(2, 2, 2) %changed from subplot(2,1,1)
                title('SNR heatmap')
                hold on
                colormap(jet(10));
                imagesc(snrs(:,:,app.StackEditField_2.Value))
                colorbar
                caxis([0 app.MaxSNREditField.Value])
                axis off
                daspect([1 1 1])
                hold off
            end
        end

        % Display data changed function: UITable
        function UITableDisplayDataChanged(app, event)
            newDisplayData = app.UITable.DisplayData;
            xlswrite("test.xls",newDisplayData);
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 939 660];
            app.UIFigure.Name = 'MATLAB App';

            % Create EdgeDetectionMethodButtonGroup
            app.EdgeDetectionMethodButtonGroup = uibuttongroup(app.UIFigure);
            app.EdgeDetectionMethodButtonGroup.Title = 'Edge Detection Method';
            app.EdgeDetectionMethodButtonGroup.Position = [49 459 130 95];

            % Create CannyButton
            app.CannyButton = uiradiobutton(app.EdgeDetectionMethodButtonGroup);
            app.CannyButton.Text = 'Canny';
            app.CannyButton.Position = [11 49 56 22];
            app.CannyButton.Value = true;

            % Create LogButton
            app.LogButton = uiradiobutton(app.EdgeDetectionMethodButtonGroup);
            app.LogButton.Text = 'Log';
            app.LogButton.Position = [11 27 42 22];

            % Create SobelButton
            app.SobelButton = uiradiobutton(app.EdgeDetectionMethodButtonGroup);
            app.SobelButton.Text = 'Sobel';
            app.SobelButton.Position = [11 5 53 22];

            % Create StackNumberEditFieldLabel
            app.StackNumberEditFieldLabel = uilabel(app.UIFigure);
            app.StackNumberEditFieldLabel.HorizontalAlignment = 'right';
            app.StackNumberEditFieldLabel.Position = [31 341 83 22];
            app.StackNumberEditFieldLabel.Text = 'Stack Number';

            % Create StackNumberEditField
            app.StackNumberEditField = uieditfield(app.UIFigure, 'numeric');
            app.StackNumberEditField.Position = [129 341 82 22];
            app.StackNumberEditField.Value = 8;

            % Create ShowMaskButton
            app.ShowMaskButton = uibutton(app.UIFigure, 'push');
            app.ShowMaskButton.ButtonPushedFcn = createCallbackFcn(app, @ShowMaskButtonPushed, true);
            app.ShowMaskButton.FontSize = 14;
            app.ShowMaskButton.Position = [49 290 100 24];
            app.ShowMaskButton.Text = 'Show Mask';

            % Create SelectImageButton
            app.SelectImageButton = uibutton(app.UIFigure, 'push');
            app.SelectImageButton.ButtonPushedFcn = createCallbackFcn(app, @SelectImageButtonPushed, true);
            app.SelectImageButton.FontSize = 14;
            app.SelectImageButton.Position = [49 597 100 24];
            app.SelectImageButton.Text = 'Select Image';

            % Create FudgeFactorEditFieldLabel
            app.FudgeFactorEditFieldLabel = uilabel(app.UIFigure);
            app.FudgeFactorEditFieldLabel.HorizontalAlignment = 'right';
            app.FudgeFactorEditFieldLabel.Position = [37 410 77 22];
            app.FudgeFactorEditFieldLabel.Text = 'Fudge Factor';

            % Create FudgeFactorEditField
            app.FudgeFactorEditField = uieditfield(app.UIFigure, 'numeric');
            app.FudgeFactorEditField.Position = [129 410 82 22];
            app.FudgeFactorEditField.Value = 1.2;

            % Create DilateEditFieldLabel
            app.DilateEditFieldLabel = uilabel(app.UIFigure);
            app.DilateEditFieldLabel.HorizontalAlignment = 'right';
            app.DilateEditFieldLabel.Position = [77 377 36 22];
            app.DilateEditFieldLabel.Text = 'Dilate';

            % Create DilateEditField
            app.DilateEditField = uieditfield(app.UIFigure, 'numeric');
            app.DilateEditField.Position = [129 377 82 22];
            app.DilateEditField.Value = 5;

            % Create MedianBackgroundSubtractCheckBox
            app.MedianBackgroundSubtractCheckBox = uicheckbox(app.UIFigure);
            app.MedianBackgroundSubtractCheckBox.Text = 'Median Background Subtract';
            app.MedianBackgroundSubtractCheckBox.Position = [30 174 183 22];

            % Create FeatureRegistrationCheckBox
            app.FeatureRegistrationCheckBox = uicheckbox(app.UIFigure);
            app.FeatureRegistrationCheckBox.Text = 'Feature Registration';
            app.FeatureRegistrationCheckBox.Position = [31 145 132 22];

            % Create GenerateHeatmapandHistogramCheckBox
            app.GenerateHeatmapandHistogramCheckBox = uicheckbox(app.UIFigure);
            app.GenerateHeatmapandHistogramCheckBox.Text = 'Generate Heatmap and Histogram';
            app.GenerateHeatmapandHistogramCheckBox.Position = [31 111 295 27];

            % Create MaxDFFEditFieldLabel
            app.MaxDFFEditFieldLabel = uilabel(app.UIFigure);
            app.MaxDFFEditFieldLabel.HorizontalAlignment = 'right';
            app.MaxDFFEditFieldLabel.Position = [34 54 55 22];
            app.MaxDFFEditFieldLabel.Text = 'Max DFF';

            % Create MaxDFFEditField
            app.MaxDFFEditField = uieditfield(app.UIFigure, 'numeric');
            app.MaxDFFEditField.Position = [104 54 45 22];
            app.MaxDFFEditField.Value = 4;

            % Create StartBeforeLigandEditFieldLabel
            app.StartBeforeLigandEditFieldLabel = uilabel(app.UIFigure);
            app.StartBeforeLigandEditFieldLabel.HorizontalAlignment = 'right';
            app.StartBeforeLigandEditFieldLabel.Position = [24 211 122 22];
            app.StartBeforeLigandEditFieldLabel.Text = 'Start (Before Ligand) ';

            % Create StartBeforeLigandEditField
            app.StartBeforeLigandEditField = uieditfield(app.UIFigure, 'numeric');
            app.StartBeforeLigandEditField.Position = [153 212 26 21];
            app.StartBeforeLigandEditField.Value = 1;

            % Create InputstacknumbersbeforeligandforaveragingLabel
            app.InputstacknumbersbeforeligandforaveragingLabel = uilabel(app.UIFigure);
            app.InputstacknumbersbeforeligandforaveragingLabel.FontSize = 14;
            app.InputstacknumbersbeforeligandforaveragingLabel.Position = [24 244 321 22];
            app.InputstacknumbersbeforeligandforaveragingLabel.Text = 'Input stack numbers before ligand (for averaging) ';

            % Create UITable
            app.UITable = uitable(app.UIFigure);
            app.UITable.ColumnName = {'StackNum'; 'MediandFF'; 'Registered'; 'Theta'};
            app.UITable.RowName = {};
            app.UITable.DisplayDataChangedFcn = createCallbackFcn(app, @UITableDisplayDataChanged, true);
            app.UITable.Position = [690 52 205 215];

            % Create StackEditField_2Label
            app.StackEditField_2Label = uilabel(app.UIFigure);
            app.StackEditField_2Label.HorizontalAlignment = 'right';
            app.StackEditField_2Label.Position = [63 85 38 22];
            app.StackEditField_2Label.Text = 'Stack';

            % Create StackEditField_2
            app.StackEditField_2 = uieditfield(app.UIFigure, 'numeric');
            app.StackEditField_2.Position = [108 86 41 21];
            app.StackEditField_2.Value = 10;

            % Create OutputTableLabel
            app.OutputTableLabel = uilabel(app.UIFigure);
            app.OutputTableLabel.FontSize = 14;
            app.OutputTableLabel.Position = [752 266 81 27];
            app.OutputTableLabel.Text = 'Output Table';

            % Create NikkiTjahjonoV12020Label
            app.NikkiTjahjonoV12020Label = uilabel(app.UIFigure);
            app.NikkiTjahjonoV12020Label.Position = [792 11 131 27];
            app.NikkiTjahjonoV12020Label.Text = 'Nikki Tjahjono V2 2021';

            % Create RunPixelWiseAnalysisButton
            app.RunPixelWiseAnalysisButton = uibutton(app.UIFigure, 'push');
            app.RunPixelWiseAnalysisButton.ButtonPushedFcn = createCallbackFcn(app, @RunPixelWiseAnalysisButtonPushed, true);
            app.RunPixelWiseAnalysisButton.FontSize = 14;
            app.RunPixelWiseAnalysisButton.Position = [194 25 166 24];
            app.RunPixelWiseAnalysisButton.Text = 'Run Pixel-Wise Analysis';

            % Create EndBeforeLigandEditFieldLabel
            app.EndBeforeLigandEditFieldLabel = uilabel(app.UIFigure);
            app.EndBeforeLigandEditFieldLabel.HorizontalAlignment = 'right';
            app.EndBeforeLigandEditFieldLabel.Position = [202 211 116 22];
            app.EndBeforeLigandEditFieldLabel.Text = 'End (Before Ligand) ';
            
            % Create EndBeforeLigandEditField
            app.EndBeforeLigandEditField = uieditfield(app.UIFigure, 'numeric');
            app.EndBeforeLigandEditField.Position = [325 212 26 21];
            app.EndBeforeLigandEditField.Value = 3;
            
            % Create MaxSNREditFieldLabel
            app.MaxSNREditFieldLabel = uilabel(app.UIFigure);
            app.MaxSNREditFieldLabel.HorizontalAlignment = 'right';
            app.MaxSNREditFieldLabel.Position = [34 26 57 22];
            app.MaxSNREditFieldLabel.Text = 'Max SNR';
            
            app.MaxSNREditField = uieditfield(app.UIFigure, 'numeric');
            app.MaxSNREditField.Position = [104 26 45 17];
            app.MaxSNREditField.Value = 40;
            
            
            % Create UIAxes
            app.UIAxes = uiaxes(app.UIFigure);
            title(app.UIAxes, 'Last Stack Here')
            xlabel(app.UIAxes, 'X')
            ylabel(app.UIAxes, 'Y')
            zlabel(app.UIAxes, 'Z')
            app.UIAxes.PlotBoxAspectRatio = [1.00389105058366 1 1];
            app.UIAxes.Position = [233 307 323 333];

            % Create UIAxes3
            app.UIAxes3 = uiaxes(app.UIFigure);
            title(app.UIAxes3, 'Output Time Trace')
            xlabel(app.UIAxes3, 'Stack Number')
            ylabel(app.UIAxes3, 'dF/F')
            zlabel(app.UIAxes3, 'Z')
            app.UIAxes3.PlotBoxAspectRatio = [1.21904761904762 1 1];
            app.UIAxes3.Position = [350 25 305 266];

            % Create UIAxes2
            app.UIAxes2 = uiaxes(app.UIFigure);
            title(app.UIAxes2, 'Mask')
            xlabel(app.UIAxes2, 'X')
            ylabel(app.UIAxes2, 'Y')
            zlabel(app.UIAxes2, 'Z')
            app.UIAxes2.PlotBoxAspectRatio = [1.00389105058366 1 1];
            app.UIAxes2.Position = [572 307 323 333];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = timeconffull_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end