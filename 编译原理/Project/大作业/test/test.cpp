#include "parser.h"
#include "gtest/gtest.h"
#include <chrono>
#include <iostream>

namespace nemo {

    // Helper function to verify parsing and measure performance
    void verify_parse(const std::string& sentence, const std::string& ans) {
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = parse(sentence, "English", 0.1, 20, 20, false, false, ReadoutMethod::FIBER_READOUT);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        double time_per_word = elapsed.count() / sentence.size();

        // Output performance metrics
        std::cout << "Sentence: \"" << sentence << "\"\n";
        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
        std::cout << "Time per word: " << time_per_word << " seconds/word\n";

        EXPECT_EQ(result, ans);
    }

    // Test case for parsing sentences
    TEST(ParserTest, BasicSentences1) {
        verify_parse("dogs are big", "are big ADJ are dogs SUBJ ");
    }
    TEST(ParserTest, BasicSentences2) {
        verify_parse("cats are bad", "are bad ADJ are cats SUBJ ");
    }
    TEST(ParserTest, BasicSentences3) {
        verify_parse("the big dogs chase the bad cats quickly", "chase quickly ADVERB chase <NON-WORD> SUBJ <NON-WORD> big ADJ <NON-WORD> the DET ");
    }
    TEST(ParserTest, BasicSentences4) {
        verify_parse("big people bite the big dogs quickly", "bite quickly ADVERB bite <NON-WORD> SUBJ <NON-WORD> big ADJ <NON-WORD> the DET ");
    }
    TEST(ParserTest, BasicSentences5) {
        verify_parse("people run", "run people SUBJ ");
    }
}  // namespace nemo

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
