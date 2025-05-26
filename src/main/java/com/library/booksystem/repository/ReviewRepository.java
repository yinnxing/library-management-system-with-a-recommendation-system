package com.library.booksystem.repository;

import com.library.booksystem.model.Review;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ReviewRepository extends JpaRepository<Review, Integer> {
    Page<Review> findByBookBookId(Integer bookId, Pageable pageable);
    Page<Review> findByUserUserId(String userId, Pageable pageable);
} 