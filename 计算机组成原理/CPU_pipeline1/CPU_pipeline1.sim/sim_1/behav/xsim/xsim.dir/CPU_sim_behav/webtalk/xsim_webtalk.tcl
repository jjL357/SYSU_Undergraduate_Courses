webtalk_init -webtalk_dir D:/CPU_pipeline1/CPU_pipeline1.sim/sim_1/behav/xsim/xsim.dir/CPU_sim_behav/webtalk/
webtalk_register_client -client project
webtalk_add_data -client project -key date_generated -value "Thu Dec  1 20:20:10 2022" -context "software_version_and_target_device"
webtalk_add_data -client project -key product_version -value "XSIM v2018.1 (64-bit)" -context "software_version_and_target_device"
webtalk_add_data -client project -key build_version -value "2188600" -context "software_version_and_target_device"
webtalk_add_data -client project -key os_platform -value "WIN64" -context "software_version_and_target_device"
webtalk_add_data -client project -key registration_id -value "" -context "software_version_and_target_device"
webtalk_add_data -client project -key tool_flow -value "xsim_vivado" -context "software_version_and_target_device"
webtalk_add_data -client project -key beta -value "FALSE" -context "software_version_and_target_device"
webtalk_add_data -client project -key route_design -value "FALSE" -context "software_version_and_target_device"
webtalk_add_data -client project -key target_family -value "not_applicable" -context "software_version_and_target_device"
webtalk_add_data -client project -key target_device -value "not_applicable" -context "software_version_and_target_device"
webtalk_add_data -client project -key target_package -value "not_applicable" -context "software_version_and_target_device"
webtalk_add_data -client project -key target_speed -value "not_applicable" -context "software_version_and_target_device"
webtalk_add_data -client project -key random_id -value "8bc6ab96-7b11-4e20-a13e-a0d4534f0fbc" -context "software_version_and_target_device"
webtalk_add_data -client project -key project_id -value "6202665dbceb478ca157a7172d1013b7" -context "software_version_and_target_device"
webtalk_add_data -client project -key project_iteration -value "323" -context "software_version_and_target_device"
webtalk_add_data -client project -key os_name -value "Microsoft Windows 8 or later , 64-bit" -context "user_environment"
webtalk_add_data -client project -key os_release -value "major release  (build 9200)" -context "user_environment"
webtalk_add_data -client project -key cpu_name -value "AMD Ryzen 5 5500U with Radeon Graphics         " -context "user_environment"
webtalk_add_data -client project -key cpu_speed -value "2096 MHz" -context "user_environment"
webtalk_add_data -client project -key total_processors -value "2" -context "user_environment"
webtalk_add_data -client project -key system_ram -value "16.000 GB" -context "user_environment"
webtalk_register_client -client xsim
webtalk_add_data -client xsim -key Command -value "xsim" -context "xsim\\command_line_options"
webtalk_add_data -client xsim -key trace_waveform -value "true" -context "xsim\\usage"
webtalk_add_data -client xsim -key runtime -value "1 us" -context "xsim\\usage"
webtalk_add_data -client xsim -key iteration -value "1" -context "xsim\\usage"
webtalk_add_data -client xsim -key Simulation_Time -value "0.06_sec" -context "xsim\\usage"
webtalk_add_data -client xsim -key Simulation_Memory -value "7248_KB" -context "xsim\\usage"
webtalk_transmit -clientid 2595529014 -regid "" -xml D:/CPU_pipeline1/CPU_pipeline1.sim/sim_1/behav/xsim/xsim.dir/CPU_sim_behav/webtalk/usage_statistics_ext_xsim.xml -html D:/CPU_pipeline1/CPU_pipeline1.sim/sim_1/behav/xsim/xsim.dir/CPU_sim_behav/webtalk/usage_statistics_ext_xsim.html -wdm D:/CPU_pipeline1/CPU_pipeline1.sim/sim_1/behav/xsim/xsim.dir/CPU_sim_behav/webtalk/usage_statistics_ext_xsim.wdm -intro "<H3>XSIM Usage Report</H3><BR>"
webtalk_terminate
