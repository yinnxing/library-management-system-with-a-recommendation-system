package com.library.booksystem.model.specification;

import com.library.booksystem.model.Book;
import com.library.booksystem.model.specification.criteria.BookCriteria;
import io.micrometer.common.lang.NonNull;
import jakarta.persistence.criteria.CriteriaBuilder;
import jakarta.persistence.criteria.CriteriaQuery;
import jakarta.persistence.criteria.Predicate;
import jakarta.persistence.criteria.Root;
import lombok.RequiredArgsConstructor;
import org.springframework.data.jpa.domain.Specification;

import java.util.ArrayList;
import java.util.List;
@RequiredArgsConstructor

public final class BookFilterSpecification implements Specification<Book> {
    private final BookCriteria criteria;

    @Override
    public Predicate toPredicate(@NonNull Root<Book> root,
                                 @NonNull CriteriaQuery<?> query,
                                 @NonNull CriteriaBuilder builder) {
        if (criteria == null) {
            return null;
        }

        List<Predicate> predicates = new ArrayList<>();

        if (criteria.getTitle() != null) {
            predicates.add(builder.like(builder.lower(root.get("title")), "%" + criteria.getTitle().toLowerCase() + "%"));
        }

        if (criteria.getAuthor() != null) {
            predicates.add(builder.like(builder.lower(root.get("author")), "%" + criteria.getAuthor().toLowerCase() + "%"));
        }

        if (criteria.getPublisher() != null) {
            predicates.add(builder.like(builder.lower(root.get("publisher")), "%" + criteria.getPublisher().toLowerCase() + "%"));
        }

        if (criteria.getPublicationYearStart() != null) {
            predicates.add(builder.greaterThanOrEqualTo(root.get("publicationYear"), criteria.getPublicationYearStart()));
        }

        if (criteria.getPublicationYearEnd() != null) {
            predicates.add(builder.lessThanOrEqualTo(root.get("publicationYear"), criteria.getPublicationYearEnd()));
        }

        if (criteria.getIsbn() != null) {
            predicates.add(builder.equal(root.get("isbn"), criteria.getIsbn()));
        }

        if (criteria.getGenre() != null) {
            predicates.add(builder.like(builder.lower(root.get("genre")), "%" + criteria.getGenre().toLowerCase() + "%"));
        }

        if (criteria.getHasCoverImage() != null) {
            if (criteria.getHasCoverImage()) {
                predicates.add(builder.isNotNull(root.get("coverImageUrl")));
            } else {
                predicates.add(builder.isNull(root.get("coverImageUrl")));
            }
        }

        if (criteria.getCreatedAtStart() != null) {
            predicates.add(builder.greaterThanOrEqualTo(root.get("createdAt"), criteria.getCreatedAtStart()));
        }

        if (criteria.getCreatedAtEnd() != null) {
            predicates.add(builder.lessThanOrEqualTo(root.get("createdAt"), criteria.getCreatedAtEnd()));
        }

        if (criteria.getQ() != null) {
            String q = "%" + criteria.getQ().toLowerCase() + "%";
            predicates.add(
                    builder.or(
                            builder.like(builder.lower(root.get("title")), q),
                            builder.like(builder.lower(root.get("author")), q),
                            builder.like(builder.lower(root.get("publisher")), q),
                            builder.like(builder.lower(root.get("isbn")), q)
                    )
            );
        }

        if (!predicates.isEmpty()) {
            query.where(predicates.toArray(new Predicate[0]));
        }

        return query.distinct(true).getRestriction();
    }
}

