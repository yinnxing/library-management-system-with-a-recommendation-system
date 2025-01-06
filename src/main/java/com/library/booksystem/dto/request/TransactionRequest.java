package com.library.booksystem.dto.request;

import com.library.booksystem.enums.TransactionStatus;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TransactionRequest {
    private String userId;
    private TransactionStatus status;
}
