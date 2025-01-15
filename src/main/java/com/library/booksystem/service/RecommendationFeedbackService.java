package com.library.booksystem.service;

import com.library.booksystem.model.RecommendationFeedback;
import com.library.booksystem.repository.RecommendationFeedbackRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class RecommendationFeedbackService {
    RecommendationFeedbackRepository recommendationFeedbackRepository;
    public void saveFeedback(RecommendationFeedback feedback) {
        recommendationFeedbackRepository.save(feedback);
    }

}
