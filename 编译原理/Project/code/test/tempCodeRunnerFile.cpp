        // 初始化自定义可塑性
        std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> custom_plasticities;
        for (const auto& area : RECURRENT_AREAS) {
            custom_plasticities["LEX"].emplace_back(area, LEX_beta);
            custom_plasticities[area].emplace_back("LEX", LEX_beta);
            custom_plasticities[area].emplace_back(area, recurrent_beta);
            for (const auto& other_area : RECURRENT_AREAS) {
                if (other_area != area) {
                    custom_plasticities[area].emplace_back(other_area, interarea_beta);
                }
            }
        }
        update_plasticities(custom_plasticities);
    }

    unordered_map<std::string, std::set<std::string>> getProjectMap(){
        unordered_map<std::string, std::set<std::string>> proj_map = ParserBrain::getProjectMap();
        if (proj_map.find("LEX") != proj_map.end() && proj_map["LEX"].size() > 2) {
            throw std::runtime_error("Got that LEX projecting into many areas: " + std::to_string(proj_map["LEX"].size()));
        }
        return proj_map;
    }

    std::string getWord(const std::string& area_name, double min_overlap = 0.7){
        std::string word = ParserBrain::getWord(area_name, min_overlap);
        if (!word.empty()) {
            return word;
        }
        if (word.empty() && area_name == "DET") {
            std::set<int> winners = area_states[area_name];
            int area_k = 100; // 假设 area_k 为 100
            double threshold = min_overlap * area_k;
            int nodet_index = DET_SIZE - 1;
            int nodet_assembly_start = nodet_index * area_k;
            std::set<int> nodet_assembly;
            for (int i = nodet_assembly_start; i < nodet_assembly_start + area_k; ++i) {
                nodet_assembly.insert(i);
            }
            std::set<int> intersection;
            std::set_intersection(winners.begin(), winners.end(), nodet_assembly.begin(), nodet_assembly.end(),
                                  std::inserter(intersection, intersection.begin()));
            if (intersection.size() > threshold) {
                return "<null-det>";