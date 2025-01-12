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
    private final TransactionMapper transactionMapper;

    public BorrowResponse borrowBook(BorrowRequest request) {
        User user = userRepository.findById(request.getUserId())
                .orElseThrow(() -> new AppException(ErrorCode.USER_NOT_EXIST));

        Book book = bookRepository.findById(request.getBookId())
                .orElseThrow(() -> new AppException(ErrorCode.BOOK_NOT_FOUND));

        if (book.getAvailableQuantity() <= 0) {
            throw new AppException(ErrorCode.BOOK_NOT_AVAILABLE);
        }

        book.setAvailableQuantity(book.getAvailableQuantity() - 1);
        bookRepository.save(book);

        Transaction transaction = new Transaction();
        transaction.setUser(user);
        transaction.setBook(book);
        transaction.setBorrowDate(LocalDateTime.now());
        transaction.setDueDate(LocalDateTime.now().plusDays(7));
        transaction.setStatus(TransactionStatus.PENDING);
        transactionRepository.save(transaction);
        return BorrowResponse.builder()
                .userId(transaction.getUser().getUserId())
                .bookId(transaction.getBook().getBookId())
                .transactionId(transaction.getTransactionId())
                .expiryDate(transaction.getDueDate().toLocalDate().toString())
                .status(transaction.getStatus().name())
                .build();
    }
//    @Scheduled(cron = "0 0 0 * * ?")
//    public void cancelPendingTransactions() {
//        LocalDateTime currentDate = LocalDateTime.now();
//
//        List<Transaction> pendingTransactions = transactionRepository.findByStatus("PENDING");
//
//        for (Transaction transaction : pendingTransactions) {
//            if (transaction.getDueDate().isBefore(currentDate)) {
//                transaction.setStatus(TransactionStatus.CANCELLED);
//
//                Book book = transaction.getBook();
//                book.setAvailableQuantity(book.getAvailableQuantity() + 1);
//                bookRepository.save(book);
//
//                transactionRepository.save(transaction);
//            }
//        }
//    }

    public Transaction returnBook(String transactionId) {
        Transaction transaction = transactionRepository.findById(transactionId)
                .orElseThrow(() -> new RuntimeException("Transaction not found"));

        if (!transaction.getStatus().equals("PENDING")) {
            throw new RuntimeException("Cannot return book for this transaction");
        }

        transaction.setStatus(TransactionStatus.RETURNED);
        transaction.setReturnDate(LocalDateTime.now());

        Book book = transaction.getBook();
        book.setAvailableQuantity(book.getAvailableQuantity() + 1);
        bookRepository.save(book);

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
    public Transaction updateTransactionStatusBorrowed(String transactionId, TransactionStatus newStatus) throws BadRequestException {
        Transaction transaction = transactionRepository.findById(transactionId)
                .orElseThrow(() -> new BadRequestException("Giao dịch không tồn tại"));
        if (transaction.getStatus() == TransactionStatus.PENDING) {
            transaction.setStatus(newStatus);
            return transactionRepository.save(transaction);
        } else {
            throw new BadRequestException("Không thể cập nhật trạng thái giao dịch này, trạng thái hiện tại không phải là PENDING");
        }
    }

    public Transaction updateTransactionStatusReturned(String transactionId, TransactionStatus newStatus) throws BadRequestException {
        Transaction transaction = transactionRepository.findById(transactionId)
                .orElseThrow(() -> new BadRequestException("Giao dịch không tồn tại"));
        if (transaction.getStatus() == TransactionStatus.BORROWED) {
            transaction.setStatus(newStatus);
            return transactionRepository.save(transaction);
        } else {
            throw new BadRequestException("Không thể cập nhật trạng thái giao dịch này, trạng thái hiện tại không phải là PENDING");
        }
    }


}
