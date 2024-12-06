package com.library.booksystem.repository;

import com.library.booksystem.model.Transaction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface TransactionRepository extends JpaRepository<Transaction, String> {
    List<Transaction> findByStatus(String status);
    @Query(value = "SELECT b.title FROM Transactions t " +
            "JOIN Books b ON t.book_id = b.book_id " +
            "WHERE t.user_id = :userId AND t.status = 'PENDING' " +
            "ORDER BY t.borrow_date DESC LIMIT 1", nativeQuery = true)
    Optional<String> findLastBorrowedBook(@Param("userId") String userId);
}
