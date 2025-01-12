package com.library.booksystem.service;

import com.library.booksystem.dto.request.WishlistRequest;
import com.library.booksystem.dto.response.BookResponse;
import com.library.booksystem.dto.response.WishlistResponse;
import com.library.booksystem.exception.AppException;
import com.library.booksystem.exception.ErrorCode;
import com.library.booksystem.mapper.BookMapper;
import com.library.booksystem.model.Book;
import com.library.booksystem.model.User;
import com.library.booksystem.model.Wishlist;
import com.library.booksystem.repository.BookRepository;
import com.library.booksystem.repository.UserRepository;
import com.library.booksystem.repository.WishlistRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class WishlistService {
    WishlistRepository wishlistRepository;
    BookRepository bookRepository;
    UserRepository userRepository;
    BookMapper bookMapper;

    public WishlistResponse addToWishlist(WishlistRequest request){
        Book book = bookRepository.findById(request.getBookId())
                .orElseThrow(() -> new AppException(ErrorCode.BOOK_NOT_FOUND));
        User user = userRepository.findById(request.getUserId())
                .orElseThrow(() -> new AppException(ErrorCode.USER_NOT_EXIST));
        Wishlist wishlist = new Wishlist();
        wishlist.setBook(book);
        wishlist.setUser(user);
        wishlistRepository.save(wishlist);
        return WishlistResponse.builder()
                .wishlistId(wishlist.getWishlistId())
                .userId(request.getUserId())
                .bookId(request.getBookId())
                .addedAt(wishlist.getAddedAt())
                .build();
    }
    public List<BookResponse> getWishlist(String userId){
        User user = userRepository.findById(userId).orElseThrow(
                () -> new AppException(ErrorCode.USER_NOT_EXIST)
        );
        List<Book> books = wishlistRepository.findWishlistByUserId(userId);
        return books.stream().map(bookMapper::toBookResponse)
                .collect(Collectors.toList());

   }
    public void removeFromWishlist(String userId, Integer bookId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new AppException(ErrorCode.USER_NOT_EXIST));
        Wishlist wishlist = wishlistRepository.findByUser_UserIdAndBook_BookId(userId, bookId);
        wishlistRepository.delete(wishlist);
    }


}
