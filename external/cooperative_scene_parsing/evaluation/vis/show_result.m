%% show our result
clc;
close all;
clear all;
addpath(genpath('.'))
igibson_colorbox = load('igibson_colorbox.mat').igibson_colorbox;
% path = 'Z:\projects\pano_3d_understanding\paper\qualitatively';
path = 'Z:\projects\pano_3d_understanding\paper\failure';
scenes = fullfile(path, 'scenes.mat');
views3d = {'bird'}; % top or bird
dosave = true;
legend = table('Size', [0, 2], 'VariableNames', {'class', 'color'}, 'VariableTypes', {'string', 'string'});

scenes = load(scenes).scenes;

%% change path
scene_names = fieldnames(scenes);
for i_scene = 1:length(scene_names)
    method_paths = scenes.(scene_names{i_scene});
    viewsCam = {};
    size_methods = size(method_paths);
    for i_method = 1:size_methods(1)
        method_name = method_paths(i_method, :);
        method_name = erase(method_name, ' ');
        if contains(method_name, 'Merom_0_int-00006') || contains(method_name, 'Merom_0_int-00086')
            rotate = 90;
        else
            rotate = 0;
        end
        disp(method_name);
        method_mat = fullfile(path, method_name);
        camera = load(method_mat);
        layout = camera.layout;
        bdb3d = camera.bdb3d;

        %% draw 3D
        f3d = figure;
        grid on
        axis equal;
        hold on;
        for kk = 1:length(bdb3d)
            b = bdb3d{kk};
            b.coeffs = b.coeffs / 2;
            vis_cube(b, igibson_colorbox(b.label + 1, :), 2);
        end
        Linewidth = 2;
        maxhight = 2;
        drawRoom(layout,'#E1B9CD',Linewidth,maxhight,'-');
%         f3d.GraphicsSmoothing = 'on';
        hold off;
        axis off;
        ax = gca;
        ax.Clipping = 'off';
%         set(gca,'CameraViewAngleMode','manual')
        for iview = 1:length(views3d)
            view3d = views3d{iview};
            if i_method == 1
                if strcmp(view3d, 'top')
                    view(rotate + 0,90);
                elseif strcmp(view3d, 'bird')
                    view(rotate - 15,45);
                end
%                 xlim(xlim + [0.5, -0.5]);
%                 ylim(ylim + [0.5, -0.5]);
%                 zlim(zlim + [0.5, -0.5]);
                xlabel('x');
                ylabel('y');
                zlabel('z');
                viewsCam{iview} = {
                    get(gca, 'CameraPosition'), get(gca, 'CameraTarget'), ...
                    xlim, ylim, zlim};
            else
%                 set(gca,'CameraViewAngleMode','manual')
                newcp = viewsCam{iview}{1};
                set(gca,'CameraPosition',newcp);
                newct = viewsCam{iview}{2};
                set(gca,'CameraTarget',newct);
                xlim(viewsCam{iview}{3});
                ylim(viewsCam{iview}{4});
                zlim(viewsCam{iview}{5});
                xlabel('x');
                ylabel('y');
                zlabel('z');
            end
            if dosave
                saveas(f3d, fullfile(path, [erase(method_name, 'data.mat') view3d '.png']));
            end
        end
    end
    if dosave
        close(f3d);
    end
    close all;
end
legend = unique(legend)

