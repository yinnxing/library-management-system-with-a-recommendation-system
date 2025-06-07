package com.library.booksystem.service;

import com.library.booksystem.enums.TransactionStatus;
import com.library.booksystem.exception.AppException;
import com.library.booksystem.exception.ErrorCode;
import com.library.booksystem.model.Transaction;
import com.library.booksystem.repository.TransactionRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.List;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
@Slf4j
public class LibraryPolicyService {
    
    // Library Policy Constants
    public static final int MAX_BORROWING_LIMIT = 5;
    public static final int MAX_RESERVATION_LIMIT = 3;
    public static final int BORROWING_PERIOD_DAYS = 1;
    public static final int HOLDING_PERIOD_DAYS = 1;
    public static final BigDecimal OVERDUE_FEE_PER_DAY = new BigDecimal("5000");
    
    TransactionRepository transactionRepository;
    EmailService emailService;
    
    /**
     * Validates if user can borrow a new book based on library policies
     */
    public void validateBorrowingEligibility(String userId) {
        // Check borrowing limit
        validateBorrowingLimit(userId);
        
        // Check for overdue books
        validateNoOverdueBooks(userId);
        
        // Check for unpaid overdue fees
        validateNoUnpaidFees(userId);
    }
    
    /**
     * Validates if user can make a new reservation
     */
    public void validateReservationEligibility(String userId) {
        // Check reservation limit
        validateReservationLimit(userId);
        
        // Check for overdue books
        validateNoOverdueBooks(userId);
        
        // Check for unpaid overdue fees
        validateNoUnpaidFees(userId);
    }
    
    /**
     * Validates borrowing limit (max 5 books)
     */
    private void validateBorrowingLimit(String userId) {
        long currentBorrowedCount = transactionRepository.countByUserIdAndStatus(userId, TransactionStatus.BORROWED);
        if (currentBorrowedCount >= MAX_BORROWING_LIMIT) {
            throw new AppException(ErrorCode.BORROWING_LIMIT_EXCEEDED);
        }
    }
    
    /**
     * Validates reservation limit (max 3 books)
     */
    private void validateReservationLimit(String userId) {
        long currentReservationCount = transactionRepository.countPendingReservationsByUserId(userId);
        if (currentReservationCount >= MAX_RESERVATION_LIMIT) {
            throw new AppException(ErrorCode.RESERVATION_LIMIT_EXCEEDED);
        }
    }
    
    /**
     * Validates user has no overdue books
     */
    private void validateNoOverdueBooks(String userId) {
        List<Transaction> overdueTransactions = transactionRepository.findOverdueTransactionsByUserId(userId, LocalDateTime.now());
        if (!overdueTransactions.isEmpty()) {
            throw new AppException(ErrorCode.OVERDUE_BOOKS_EXIST);
        }
    }
    
    /**
     * Validates user has no unpaid overdue fees
     */
    private void validateNoUnpaidFees(String userId) {
        BigDecimal unpaidFees = transactionRepository.calculateUnpaidOverdueFees(userId);
        if (unpaidFees.compareTo(BigDecimal.ZERO) > 0) {
            throw new AppException(ErrorCode.UNPAID_OVERDUE_FEES);
        }
    }
    
    /**
     * Sets pickup deadline for a pending transaction (3 days from borrow date)
     */
    public void setPickupDeadline(Transaction transaction) {
        LocalDateTime pickupDeadline = transaction.getBorrowDate().plusDays(HOLDING_PERIOD_DAYS);
        transaction.setPickupDeadline(pickupDeadline);
    }
    
    /**
     * Sets due date for a borrowed transaction (14 days from borrow date)
     */
    public void setDueDate(Transaction transaction) {
        LocalDateTime dueDate = LocalDateTime.now().plusDays(BORROWING_PERIOD_DAYS);
        transaction.setDueDate(dueDate);
    }
    
    /**
     * Calculates overdue fee for a transaction
     */
    public BigDecimal calculateOverdueFee(Transaction transaction) {
        if (transaction.getDueDate() == null || transaction.getDueDate().isAfter(LocalDateTime.now())) {
            return BigDecimal.ZERO;
        }
        
        long overdueDays = ChronoUnit.DAYS.between(transaction.getDueDate(), LocalDateTime.now());
        return OVERDUE_FEE_PER_DAY.multiply(BigDecimal.valueOf(overdueDays));
    }
    
    /**
     * Updates overdue fee for a transaction
     */
    public void updateOverdueFee(Transaction transaction) {
        BigDecimal overdueFee = calculateOverdueFee(transaction);
        transaction.setOverdueFee(overdueFee);
    }
    
    /**
     * Scheduled task to cancel expired pending transactions (runs daily at midnight)
     */
    @Scheduled(cron = "0 0 0 * * ?")
    public void cancelExpiredPendingTransactions() {
        LocalDateTime currentDate = LocalDateTime.now();
        List<Transaction> expiredTransactions = transactionRepository.findExpiredPendingTransactions(currentDate);
        
        for (Transaction transaction : expiredTransactions) {
            transaction.setStatus(TransactionStatus.CANCELLED);
            transactionRepository.save(transaction);
            log.info("Cancelled expired pending transaction: {}", transaction.getTransactionId());
        }
    }
    
    /**
     * Scheduled task to update overdue fees and mark overdue transactions (runs daily at midnight)
     */
    @Scheduled(cron = "0 0 0 * * ?")
    public void updateOverdueTransactions() {
        LocalDateTime currentDate = LocalDateTime.now();
        List<Transaction> overdueTransactions = transactionRepository.findAllOverdueTransactions(currentDate);
        
        for (Transaction transaction : overdueTransactions) {
            // Update status to OVERDUE
            transaction.setStatus(TransactionStatus.OVERDUE);
            
            // Calculate and update overdue fee
            updateOverdueFee(transaction);
            
            transactionRepository.save(transaction);
            
            // Send overdue notification email
            emailService.sendOverdueNotification(transaction);
            
            log.info("Updated overdue transaction: {} with fee: {}", 
                    transaction.getTransactionId(), transaction.getOverdueFee());
        }
    }
    
    /**
     * Scheduled task to send return reminder emails (runs daily at 9 AM)
     * Sends reminders 3 days before due date
     */
    @Scheduled(cron = "0 0 9 * * ?")
    public void sendReturnReminders() {
        LocalDateTime reminderDate = LocalDateTime.now().plusDays(3);
        List<Transaction> transactionsNearDue = transactionRepository.findTransactionsDueOnDate(reminderDate);
        
        for (Transaction transaction : transactionsNearDue) {
            if (transaction.getStatus() == TransactionStatus.BORROWED) {
                emailService.sendReturnReminderNotification(transaction);
                log.info("Sent return reminder for transaction: {}", transaction.getTransactionId());
            }
        }
        
        log.info("Processed {} return reminders", transactionsNearDue.size());
    }
    
    /**
     * Gets user's borrowing statistics
     */
    public BorrowingStats getUserBorrowingStats(String userId) {
        long borrowedCount = transactionRepository.countByUserIdAndStatus(userId, TransactionStatus.BORROWED);
        long pendingCount = transactionRepository.countPendingReservationsByUserId(userId);
        List<Transaction> overdueTransactions = transactionRepository.findOverdueTransactionsByUserId(userId, LocalDateTime.now());
        BigDecimal unpaidFees = transactionRepository.calculateUnpaidOverdueFees(userId);
        
        return BorrowingStats.builder()
                .currentBorrowedBooks((int) borrowedCount)
                .currentReservations((int) pendingCount)
                .overdueBooks(overdueTransactions.size())
                .unpaidOverdueFees(unpaidFees)
                .maxBorrowingLimit(MAX_BORROWING_LIMIT)
                .maxReservationLimit(MAX_RESERVATION_LIMIT)
                .build();
    }
    
    /**
     * Inner class for borrowing statistics
     */
    @lombok.Data
    @lombok.Builder
    public static class BorrowingStats {
        private int currentBorrowedBooks;
        private int currentReservations;
        private int overdueBooks;
        private BigDecimal unpaidOverdueFees;
        private int maxBorrowingLimit;
        private int maxReservationLimit;
    }
} 