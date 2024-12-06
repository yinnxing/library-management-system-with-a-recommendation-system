package com.library.booksystem.service;

import com.library.booksystem.dto.request.BorrowRequest;
import com.library.booksystem.dto.response.TransactionResponse;
import com.library.booksystem.enums.TransactionStatus;
import com.library.booksystem.exception.AppException;
import com.library.booksystem.exception.ErrorCode;
import com.library.booksystem.model.Book;
import com.library.booksystem.model.Transaction;
import com.library.booksystem.model.User;
import com.library.booksystem.repository.BookRepository;
import com.library.booksystem.repository.TransactionRepository;
import com.library.booksystem.repository.UserRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class TransactionService {
    TransactionRepository transactionRepository;
    UserRepository userRepository;
    BookRepository bookRepository;
    public TransactionResponse borrowBook(BorrowRequest request) {
        User user = userRepository.findById(request.getUserId())
                .orElseThrow(() -> new AppException(ErrorCode.USER_NOT_EXIST));

        Book book = bookRepository.findById(request.getBookId())
                .orElseThrow(() -> new AppException(ErrorCode.BOOK_NOT_FOUND));

        if (book.getAvailableQuantity() <= 0) {
            throw new AppException(ErrorCode.BOOK_NOT_AVAILABLE);
        }

        // Cập nhật số lượng sách có sẵn
        book.setAvailableQuantity(book.getAvailableQuantity() - 1);
        bookRepository.save(book);

        // Tạo giao dịch mượn sách
        Transaction transaction = new Transaction();
        transaction.setUser(user);
        transaction.setBook(book);
        transaction.setBorrowDate(LocalDateTime.now());
        transaction.setDueDate(LocalDateTime.now().plusDays(7));  // 7 ngày sau
        transaction.setStatus(TransactionStatus.PENDING);  // Trạng thái ban đầu là 'PENDING'
        transactionRepository.save(transaction);
        return TransactionResponse.builder()
                .userId(transaction.getUser().getUserId())
                .bookId(transaction.getBook().getBookId())
                .transactionId(transaction.getTransactionId())
                .expiryDate(transaction.getDueDate().toLocalDate().toString()) // Ngày hết hạn
                .status(transaction.getStatus().name())
                .build();
    }
    // Xử lý khi người dùng không mượn sách trong vòng 7 ngày
    @Scheduled(cron = "0 0 0 * * ?")  // Chạy mỗi ngày vào lúc nửa đêm
    public void cancelPendingTransactions() {
        LocalDateTime currentDate = LocalDateTime.now();

        List<Transaction> pendingTransactions = transactionRepository.findByStatus("PENDING");

        for (Transaction transaction : pendingTransactions) {
            if (transaction.getDueDate().isBefore(currentDate)) {
                // Hủy giao dịch quá hạn
                transaction.setStatus(TransactionStatus.CANCELLED);

                // Khôi phục lại số lượng sách có sẵn
                Book book = transaction.getBook();
                book.setAvailableQuantity(book.getAvailableQuantity() + 1);
                bookRepository.save(book);

                transactionRepository.save(transaction);
            }
        }
    }

    // Xử lý khi người dùng thực hiện trả sách offline
    public Transaction returnBook(String transactionId) {
        Transaction transaction = transactionRepository.findById(transactionId)
                .orElseThrow(() -> new RuntimeException("Transaction not found"));

        if (!transaction.getStatus().equals("PENDING")) {
            throw new RuntimeException("Cannot return book for this transaction");
        }

        // Cập nhật trạng thái giao dịch thành 'RETURNED'
        transaction.setStatus(TransactionStatus.RETURNED);
        transaction.setReturnDate(LocalDateTime.now());

        // Cập nhật số lượng sách có sẵn
        Book book = transaction.getBook();
        book.setAvailableQuantity(book.getAvailableQuantity() + 1);
        bookRepository.save(book);

        return transactionRepository.save(transaction);
    }

}
