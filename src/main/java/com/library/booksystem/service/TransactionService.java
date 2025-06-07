package com.library.booksystem.service;

import com.library.booksystem.dto.request.BorrowRequest;
import com.library.booksystem.dto.request.TransactionRequest;
import com.library.booksystem.dto.response.BorrowResponse;
import com.library.booksystem.dto.response.TransactionResponse;
import com.library.booksystem.enums.TransactionStatus;
import com.library.booksystem.exception.AppException;
import com.library.booksystem.exception.ErrorCode;
import com.library.booksystem.mapper.TransactionMapper;
import com.library.booksystem.model.Book;
import com.library.booksystem.model.Transaction;
import com.library.booksystem.model.User;
import com.library.booksystem.model.specification.TransactionFilterSpecification;
import com.library.booksystem.model.specification.criteria.PaginationCriteria;
import com.library.booksystem.model.specification.criteria.TransactionCriteria;
import com.library.booksystem.repository.BookRepository;
import com.library.booksystem.repository.TransactionRepository;
import com.library.booksystem.repository.UserRepository;
import com.library.booksystem.util.PageRequestBuilder;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.apache.coyote.BadRequestException;
import org.springframework.data.domain.Page;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class TransactionService {
    TransactionRepository transactionRepository;
    UserRepository userRepository;
    BookRepository bookRepository;
    LibraryPolicyService libraryPolicyService;
    EmailService emailService;
    private final TransactionMapper transactionMapper;

    public BorrowResponse borrowBook(BorrowRequest request) {
        User user = userRepository.findById(request.getUserId())
                .orElseThrow(() -> new AppException(ErrorCode.USER_NOT_EXIST));

        Book book = bookRepository.findById(request.getBookId())
                .orElseThrow(() -> new AppException(ErrorCode.BOOK_NOT_FOUND));

        if (book.getAvailableQuantity() <= 0) {
            throw new AppException(ErrorCode.BOOK_NOT_AVAILABLE);
        }

        // Validate library policies before borrowing
        libraryPolicyService.validateReservationEligibility(request.getUserId());

        // Create transaction
        Transaction transaction = new Transaction();
        transaction.setUser(user);
        transaction.setBook(book);
        transaction.setBorrowDate(LocalDateTime.now());
        transaction.setStatus(TransactionStatus.PENDING);
        
        // Set pickup deadline (3 days from borrow date)
        libraryPolicyService.setPickupDeadline(transaction);
        
        // Save transaction first to get the ID
        transaction = transactionRepository.save(transaction);
        
        // Update book availability after successful transaction creation
        book.setAvailableQuantity(book.getAvailableQuantity() - 1);
        bookRepository.save(book);

        return BorrowResponse.builder()
                .userId(transaction.getUser().getUserId())
                .bookId(transaction.getBook().getBookId())
                .transactionId(transaction.getTransactionId())
                .bookTitle(transaction.getBook().getTitle())
                .borrowDate(transaction.getBorrowDate())
                .pickupDeadline(transaction.getPickupDeadline())
                .status(transaction.getStatus().name())
                .message("Book reserved successfully. Please pick up within 3 days.")
                .build();
    }

    public Transaction returnBook(String transactionId) {
        Transaction transaction = transactionRepository.findById(transactionId)
                .orElseThrow(() -> new RuntimeException("Transaction not found"));

        if (!transaction.getStatus().equals(TransactionStatus.BORROWED) && 
            !transaction.getStatus().equals(TransactionStatus.OVERDUE)) {
            throw new RuntimeException("Cannot return book for this transaction");
        }

        // Calculate final overdue fee before returning
        if (transaction.getStatus().equals(TransactionStatus.OVERDUE)) {
            libraryPolicyService.updateOverdueFee(transaction);
        }

        transaction.setStatus(TransactionStatus.RETURNED);
        transaction.setReturnDate(LocalDateTime.now());

        // Update book availability
        Book book = transaction.getBook();
        book.setAvailableQuantity(book.getAvailableQuantity() + 1);
        bookRepository.save(book);

        // Save and return the updated transaction
        return transactionRepository.save(transaction);
    }

    public Page<TransactionResponse> getTransactions(TransactionCriteria transactionCriteria, PaginationCriteria paginationCriteria) throws BadRequestException {
        Page<Transaction> transactions = transactionRepository.findAll(new TransactionFilterSpecification(transactionCriteria), PageRequestBuilder.build(paginationCriteria));
        return transactions.map(transactionMapper::toTransactionResponse);
    }
    
    public List<TransactionResponse> getTransactions(TransactionRequest request) {
        List<Transaction> transactions = transactionRepository.findByUser_UserIdAndStatus(request.getUserId(), request.getStatus());
        return transactions.stream().map(transactionMapper::toTransactionResponse).collect(Collectors.toList());
    }
    
    public TransactionResponse updateTransactionStatusBorrowed(String transactionId, TransactionStatus newStatus) throws BadRequestException {
        Transaction transaction = transactionRepository.findById(transactionId)
                .orElseThrow(() -> new BadRequestException("Giao dịch không tồn tại"));
        
        if (transaction.getStatus() == TransactionStatus.PENDING) {
            // Validate borrowing eligibility when converting from PENDING to BORROWED
            libraryPolicyService.validateBorrowingEligibility(transaction.getUser().getUserId());
            
            transaction.setStatus(newStatus);
            // Set due date (14 days from now)
            libraryPolicyService.setDueDate(transaction);
            
            // Save the updated transaction
            transaction = transactionRepository.save(transaction);
            
            // Send email notification when book is successfully borrowed
            if (newStatus == TransactionStatus.BORROWED) {
                emailService.sendBookBorrowedNotification(transaction);
            }
            
            return transactionMapper.toTransactionResponse(transaction);
        } else {
            throw new BadRequestException("Không thể cập nhật trạng thái giao dịch này, trạng thái hiện tại không phải là PENDING");
        }
    }

    public TransactionResponse updateTransactionStatusReturned(String transactionId, TransactionStatus newStatus) throws BadRequestException {
        Transaction transaction = transactionRepository.findById(transactionId)
                .orElseThrow(() -> new BadRequestException("Giao dịch không tồn tại"));
        
        if (transaction.getStatus() == TransactionStatus.BORROWED || transaction.getStatus() == TransactionStatus.OVERDUE) {
            // Calculate final overdue fee before returning
            if (transaction.getStatus() == TransactionStatus.OVERDUE) {
                libraryPolicyService.updateOverdueFee(transaction);
            }
            
            transaction.setStatus(newStatus);
            transaction.setReturnDate(LocalDateTime.now());
            
            // Update book availability
            Book book = transaction.getBook();
            book.setAvailableQuantity(book.getAvailableQuantity() + 1);
            bookRepository.save(book);
            
            // Save the updated transaction
            transaction = transactionRepository.save(transaction);
            return transactionMapper.toTransactionResponse(transaction);
        } else {
            throw new BadRequestException("Không thể cập nhật trạng thái giao dịch này, trạng thái hiện tại không phải là BORROWED hoặc OVERDUE");
        }
    }
    
    /**
     * Get user's borrowing statistics
     */
    public LibraryPolicyService.BorrowingStats getUserBorrowingStats(String userId) {
        return libraryPolicyService.getUserBorrowingStats(userId);
    }
    
    /**
     * Pay overdue fees for a user
     */
    public void payOverdueFees(String userId, BigDecimal amount) {
        List<Transaction> overdueTransactions = transactionRepository.findOverdueTransactionsByUserId(userId, LocalDateTime.now());
        
        BigDecimal remainingAmount = amount;
        for (Transaction transaction : overdueTransactions) {
            if (remainingAmount.compareTo(BigDecimal.ZERO) <= 0) {
                break;
            }
            
            BigDecimal transactionFee = transaction.getOverdueFee();
            if (transactionFee.compareTo(BigDecimal.ZERO) > 0) {
                if (remainingAmount.compareTo(transactionFee) >= 0) {
                    // Pay full fee for this transaction
                    remainingAmount = remainingAmount.subtract(transactionFee);
                    transaction.setOverdueFee(BigDecimal.ZERO);
                } else {
                    // Partial payment
                    transaction.setOverdueFee(transactionFee.subtract(remainingAmount));
                    remainingAmount = BigDecimal.ZERO;
                }
                transactionRepository.save(transaction);
            }
        }
    }
    
    /**
     * Cancel a pending transaction
     */
    public TransactionResponse cancelPendingTransaction(String transactionId) throws BadRequestException {
        Transaction transaction = transactionRepository.findById(transactionId)
                .orElseThrow(() -> new BadRequestException("Giao dịch không tồn tại"));
        
        if (transaction.getStatus() != TransactionStatus.PENDING) {
            throw new BadRequestException("Chỉ có thể hủy giao dịch đang chờ xử lý");
        }
        
        transaction.setStatus(TransactionStatus.CANCELLED);
        
        // Return book to available quantity
        Book book = transaction.getBook();
        book.setAvailableQuantity(book.getAvailableQuantity() + 1);
        bookRepository.save(book);
        
        // Save the updated transaction
        transaction = transactionRepository.save(transaction);
        return transactionMapper.toTransactionResponse(transaction);
    }
}
