package com.library.booksystem.service;

import com.library.booksystem.dto.request.BookRequest;
import com.library.booksystem.dto.response.BookResponse;
import com.library.booksystem.exception.AppException;
import com.library.booksystem.exception.ErrorCode;
import com.library.booksystem.mapper.BookMapper;
import com.library.booksystem.model.Book;
import com.library.booksystem.model.specification.BookFilterSpecification;
import com.library.booksystem.model.specification.criteria.BookCriteria;
import com.library.booksystem.model.specification.criteria.PaginationCriteria;
import com.library.booksystem.repository.BookRepository;
import com.library.booksystem.repository.TransactionRepository;
import com.library.booksystem.util.PageRequestBuilder;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.apache.coyote.BadRequestException;
import org.springframework.data.domain.Page;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;


@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class BookService {
    BookRepository bookRepository;
    BookMapper bookMapper;
    TransactionRepository transactionRepository;

    public Page<BookResponse> getBooks(BookCriteria bookCriteria, PaginationCriteria paginationCriteria) throws BadRequestException {
        Page<Book> books = bookRepository.findAll(new BookFilterSpecification(bookCriteria), PageRequestBuilder.build(paginationCriteria));
        Page<BookResponse> bookResponses = books.map(bookMapper::toBookResponse);
        return bookResponses;
    }

    public BookResponse getBook(Integer bookId){
        Book book = bookRepository.findById(bookId).orElseThrow(
                () -> new AppException(ErrorCode.BOOK_NOT_FOUND)
        );
        return bookMapper.toBookResponse(book);
    }

    public BookResponse createBook(BookRequest request){
        Book book = bookMapper.toBook(request);
        book.setCreatedAt(LocalDateTime.now());//
        bookRepository.save(book);
        return bookMapper.toBookResponse(book);
    }
    public BookResponse updateBook(Integer bookId, BookRequest request) {
        Book existingBook = bookRepository.findById(bookId).orElseThrow(
                () -> new AppException(ErrorCode.BOOK_NOT_FOUND)
        );
        bookMapper.updateBook(existingBook, request);
        bookRepository.save(existingBook);
        return bookMapper.toBookResponse(existingBook);
    }

    public void deleteBook(Integer bookId) {
        Book book = bookRepository.findById(bookId).orElseThrow(
                () -> new AppException(ErrorCode.BOOK_NOT_FOUND)
        );
        bookRepository.delete(book);
    }

    public String getLastBorrowedBook(String userId) {
        return transactionRepository.findLastBorrowedBook(userId)
                .orElseThrow(() -> new RuntimeException("Không tìm thấy sách mượn cuối cùng"));
    }

}
