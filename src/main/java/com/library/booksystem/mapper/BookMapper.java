package com.library.booksystem.mapper;

import com.library.booksystem.dto.request.BookRequest;
import com.library.booksystem.dto.response.BookResponse;
import com.library.booksystem.model.Book;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.MappingTarget;

@Mapper(componentModel = "spring")
public interface BookMapper {
    BookResponse toBookResponse(Book book);
    Book toBook(BookRequest request);
    void updateBook(@MappingTarget Book book, BookRequest request);


}
