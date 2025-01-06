package com.library.booksystem.mapper;

import com.library.booksystem.dto.response.TransactionResponse;
import com.library.booksystem.model.Transaction;
import org.mapstruct.Mapper;

@Mapper(componentModel = "spring")

public interface TransactionMapper {
    TransactionResponse toTransactionResponse(Transaction transaction);
}
