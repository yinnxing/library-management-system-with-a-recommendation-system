package com.library.booksystem.controller;

import com.library.booksystem.dto.request.TransactionRequest;
import com.library.booksystem.dto.response.TransactionResponse;
import com.library.booksystem.service.TransactionService;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
@RequiredArgsConstructor
@RequestMapping("/transactions")
public class TransactionController {
    TransactionService transactionService;
    @PostMapping("/search")
    public List<TransactionResponse> getTransactions(@RequestBody TransactionRequest transactionRequest) {
        return transactionService.getTransactions(transactionRequest);
    }


}
