package com.library.booksystem.controller;

import com.library.booksystem.dto.request.BorrowRequest;
import com.library.booksystem.dto.request.WishlistRequest;
import com.library.booksystem.dto.response.ApiResponse;
import com.library.booksystem.dto.response.BookResponse;
import com.library.booksystem.dto.response.BorrowResponse;
import com.library.booksystem.dto.response.WishlistResponse;
import com.library.booksystem.model.specification.criteria.BookCriteria;
import com.library.booksystem.model.specification.criteria.PaginationCriteria;
import com.library.booksystem.service.BookService;
import com.library.booksystem.service.TransactionService;
import com.library.booksystem.service.WishlistService;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.apache.coyote.BadRequestException;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
@RequiredArgsConstructor
@RequestMapping("/books")
public class BookController {
    BookService bookService;
    TransactionService transactionService;
    WishlistService wishlistService;
    @GetMapping
    ApiResponse<Page<BookResponse>> getBooks(BookCriteria bookCriteria, PaginationCriteria paginationCriteria) throws BadRequestException {
        return ApiResponse.<Page<BookResponse>>builder()
                .result(bookService.getBooks(bookCriteria, paginationCriteria))
                .build();
    }
    @GetMapping("/{bookId}")
    ApiResponse<BookResponse> getBook(@PathVariable Integer bookId){
        return ApiResponse.<BookResponse>builder()
                .result(bookService.getBook(bookId))
                .build();
    }
//    @PostMapping
//    ApiResponse<BookResponse> createBook(@RequestBody BookRequest request){
//        return ApiResponse.<BookResponse>builder()
//                .result(bookService.createBook(request))
//                .build();
//    }

    @PostMapping("/borrow")
    ApiResponse<BorrowResponse> borrowBook(@RequestBody BorrowRequest request){
        return ApiResponse.<BorrowResponse>builder()
                .result(transactionService.borrowBook(request))
                .build();
    }
    @PostMapping("/wishlist")
    ApiResponse<WishlistResponse> addToWishlist(@RequestBody WishlistRequest request){
        return ApiResponse.<WishlistResponse>builder()
                .result(wishlistService.addToWishlist(request))
                .build();
    }

    @GetMapping("/wishlist")
    ApiResponse<List<BookResponse>> getMyWishlist(@RequestParam String userId){
        return ApiResponse.<List<BookResponse>>builder()
                .result(wishlistService.getWishlist(userId))
                .build();
    }
    @DeleteMapping("/wishlist")
    ApiResponse<Void>removeFromWishlist(@RequestBody WishlistRequest request) {
        wishlistService.removeFromWishlist(request.getUserId(), request.getBookId());
        return ApiResponse.<Void>builder().build();
}





    

}
