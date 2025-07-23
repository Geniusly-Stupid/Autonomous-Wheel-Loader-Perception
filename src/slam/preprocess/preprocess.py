def generate_files_and_trajectory(folder_name="rgb", 
                                   start_timestamp=0.0, 
                                   interval=0.1, 
                                   total_images=1000, 
                                   output_image_file=r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\sequence1\rgb.txt", 
                                   output_trajectory_file=r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\sequence1\trajectory.txt",
                                    type_name="jpg"):
    # Generate timestamps at 0.1-second intervals
    timestamps = [f"{start_timestamp + i * interval:.6f}" for i in range(total_images)]

    # Write the filenames with timestamps for images
    with open(output_image_file, "w") as f:
        for i, timestamp in enumerate(timestamps, start=1):
            filename = f"{folder_name}\\{folder_name}_{i:04d}.{type_name}"
            f.write(f"{timestamp} {filename}\n")
    
    # Write the ground truth trajectory (stationary pose)
    with open(output_trajectory_file, "w") as f:
        f.write("# ground truth trajectory\n")
        f.write("# file: 'slam'\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        
        # Write the stationary pose for each timestamp
        for timestamp in timestamps:
            tx, ty, tz = 0.0, 0.0, 0.0  # Stationary position
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0  # Identity quaternion
            f.write(f"{timestamp} {tx:.18e} {ty:.18e} {tz:.18e} {qx:.18e} {qy:.18e} {qz:.18e} {qw:.18e}\n")

    print(f"Output written to {output_image_file} and {output_trajectory_file}")

if __name__ == "__main__":
    generate_files_and_trajectory(folder_name="color", total_images=390, type_name="jpg")
    generate_files_and_trajectory(folder_name="depth", total_images=390, 
                                   output_image_file=r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\sequence1\depth.txt", 
                                   output_trajectory_file=r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\sequence1\groundtruth.txt", type_name="png")
