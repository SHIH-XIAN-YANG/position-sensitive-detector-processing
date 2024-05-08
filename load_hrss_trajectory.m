
path_dir = "points.txt";
try
  % Read data using dlmread for flexibility with delimiters and data types
  data =  readmatrix(path_dir, 'Delimiter', ',');


  % Extract and scale data columns (assuming consistent data types)
  x_c = data(:, 1) / 1000000;
  y_c = data(:, 2) / 1000000;
  z_c = data(:, 3) / 1000000;
  pitch_c = data(:, 4) / 1000;
  yaw_c = data(:, 5) / 1000;
  roll_c = data(:, 6) / 1000;
  q1_c = data(:, 7) / 1000;
  q2_c = data(:, 8) / 1000;
  q3_c = data(:, 9) / 1000;
  q4_c = data(:, 10) / 1000;
  q5_c = data(:, 11) / 1000;
  q6_c = data(:, 12) / 1000;
catch
  % Handle errors gracefully (e.g., return empty outputs or log error)
  x_c = [];
  y_c = [];
  z_c = [];
  pitch_c = [];
  yaw_c = [];
  roll_c = [];
  q1_c = [];
  q2_c = [];
  q3_c = [];
  q4_c = [];
  q5_c = [];
  q6_c = [];
  path_mode = -1;  % Indicate error (consider a more informative value)
  warning('Error loading HRSS trajectory');
end
