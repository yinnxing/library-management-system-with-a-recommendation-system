package com.library.booksystem.repository;

import com.library.booksystem.model.RecommendationFeedback;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface RecommendationFeedbackRepository extends JpaRepository<RecommendationFeedback,Long> {
}
