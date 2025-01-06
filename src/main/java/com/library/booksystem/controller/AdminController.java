package com.library.booksystem.controller;

import com.library.booksystem.dto.request.BookRequest;
import com.library.booksystem.dto.response.ApiResponse;
import com.library.booksystem.dto.response.BookResponse;
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
    public ApiResponse<Page<Transaction>> getTransactions(TransactionCriteria transactionCriteria, PaginationCriteria paginationCriteria) throws BadRequestException {
        return ApiResponse.<Page<Transaction>>builder()
                .result(transactionService.getTransactions(transactionCriteria, paginationCriteria))
                .build();
    }
    @PostMapping("/book")
    public ApiResponse<BookResponse> createBook(@RequestBody BookRequest request){
        return ApiResponse.<BookResponse>builder()
                .result(bookService.createBook(request))
                .build();
    }




}
