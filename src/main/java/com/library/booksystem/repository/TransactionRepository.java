package com.library.booksystem.repository;

import com.library.booksystem.enums.TransactionStatus;
import com.library.booksystem.model.Transaction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface TransactionRepository extends JpaRepository<Transaction, String>, JpaSpecificationExecutor<Transaction> {
    List<Transaction> findByStatus(String status);
    List<Transaction> findByUser_UserIdAndStatus(String userId, TransactionStatus status);

    @Query(value = "SELECT b.title FROM Transactions t " +
            "JOIN Books b ON t.book_id = b.book_id " +
            "WHERE t.user_id = :userId AND t.status = 'PENDING' " +
            "ORDER BY t.borrow_date DESC LIMIT 1", nativeQuery = true)
    Optional<String> findLastBorrowedBook(@Param("userId") String userId);
    
    // Count currently borrowed books (BORROWED status)
    @Query("SELECT COUNT(t) FROM Transaction t WHERE t.user.userId = :userId AND t.status = :status")
    long countByUserIdAndStatus(@Param("userId") String userId, @Param("status") TransactionStatus status);
    
    // Count pending reservations (PENDING status)
    @Query("SELECT COUNT(t) FROM Transaction t WHERE t.user.userId = :userId AND t.status = 'PENDING'")
    long countPendingReservationsByUserId(@Param("userId") String userId);
    
    // Find overdue transactions (BORROWED status with due date passed)
    @Query("SELECT t FROM Transaction t WHERE t.user.userId = :userId AND t.status = 'BORROWED' AND t.dueDate < :currentDate")
    List<Transaction> findOverdueTransactionsByUserId(@Param("userId") String userId, @Param("currentDate") LocalDateTime currentDate);
    
    // Find expired pending transactions (PENDING status with pickup deadline passed)
    @Query("SELECT t FROM Transaction t WHERE t.status = 'PENDING' AND t.pickupDeadline < :currentDate")
    List<Transaction> findExpiredPendingTransactions(@Param("currentDate") LocalDateTime currentDate);
    
    // Calculate total unpaid overdue fees for a user
    @Query("SELECT COALESCE(SUM(t.overdueFee), 0) FROM Transaction t WHERE t.user.userId = :userId AND t.overdueFee > 0 AND t.status IN ('BORROWED', 'OVERDUE')")
    BigDecimal calculateUnpaidOverdueFees(@Param("userId") String userId);
    
    // Find all overdue transactions that need fee calculation
    @Query("SELECT t FROM Transaction t WHERE t.status = 'BORROWED' AND t.dueDate < :currentDate")
    List<Transaction> findAllOverdueTransactions(@Param("currentDate") LocalDateTime currentDate);
    
    // Find transactions due on a specific date (for return reminders)
    @Query("SELECT t FROM Transaction t WHERE t.status = 'BORROWED' AND DATE(t.dueDate) = DATE(:dueDate)")
    List<Transaction> findTransactionsDueOnDate(@Param("dueDate") LocalDateTime dueDate);
}
