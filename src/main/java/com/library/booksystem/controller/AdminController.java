package com.library.booksystem.controller;

import com.library.booksystem.dto.request.BookRequest;
import com.library.booksystem.dto.response.ApiResponse;
import com.library.booksystem.dto.response.BookResponse;
import com.library.booksystem.dto.response.TransactionResponse;
import com.library.booksystem.enums.TransactionStatus;
import com.library.booksystem.model.Transaction;
import com.library.booksystem.model.specification.criteria.PaginationCriteria;
import com.library.booksystem.model.specification.criteria.TransactionCriteria;
import com.library.booksystem.service.BookService;
import com.library.booksystem.service.TransactionService;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.apache.coyote.BadRequestException;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/admin")
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class AdminController {
    TransactionService transactionService;
    BookService bookService;
    @GetMapping("/transactions")
    public ApiResponse<Page<TransactionResponse>> getTransactions(TransactionCriteria transactionCriteria, PaginationCriteria paginationCriteria) throws BadRequestException {
        return ApiResponse.<Page<TransactionResponse>>builder()
                .result(transactionService.getTransactions(transactionCriteria, paginationCriteria))
                .build();
    }
    @PostMapping("/book")
    public ApiResponse<BookResponse> createBook(@RequestBody BookRequest request){
        return ApiResponse.<BookResponse>builder()
                .result(bookService.createBook(request))
                .build();
    }
    /////
    @PutMapping("/{bookId}")
    ApiResponse<BookResponse> updateBook(@PathVariable Integer bookId, @RequestBody BookRequest request) {
        return ApiResponse.<BookResponse>builder()
                .result(bookService.updateBook(bookId, request))
                .build();
    }

    @DeleteMapping("book/{bookId}")
    ApiResponse<Void> deleteBook(@PathVariable Integer bookId) {
        bookService.deleteBook(bookId);
        return ApiResponse.<Void>builder()
                .message("Sách đã được xóa thành công")
                .build();
    }
    @PutMapping("/transactions/{transactionId}/update-borrowed")
    public ApiResponse<Transaction> updateTransactionStatus(
            @PathVariable String transactionId) throws BadRequestException {
        Transaction updatedTransaction = transactionService.updateTransactionStatusBorrowed(transactionId, TransactionStatus.BORROWED);
        return ApiResponse.<Transaction>builder()
                .result(updatedTransaction)
                .message("Trạng thái giao dịch đã được cập nhật thành công")
                .build();
    }
    @PutMapping("/transactions/{transactionId}/update-returned")
    public ApiResponse<Transaction> updateToReturnedTransactionStatus(
            @PathVariable String transactionId) throws BadRequestException {
        Transaction updatedTransaction = transactionService.updateTransactionStatusReturned(transactionId, TransactionStatus.RETURNED);
        return ApiResponse.<Transaction>builder()
                .result(updatedTransaction)
                .message("Trạng thái giao dịch đã được cập nhật thành công")
                .build();
    }





}
